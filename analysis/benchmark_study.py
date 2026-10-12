#!/usr/bin/env python3
"""Standardized GPU forward latency and fixed-input resource metrics for 14 models."""

import argparse
import fcntl
import gc
import hashlib
import math
import linecache
import os
import random
import statistics
import time
import traceback
import warnings
from pathlib import Path
from contextlib import ExitStack
from collections import Counter

import torch

from analysis.quantization_report import (infer_model_input_dtype,
                                         replace_int8_supported_layers, state_dict_numel)
from experiments.run_study import (read_json, sha256, write_json,
                                   assert_idle, evaluation_data, gpu_uuid)
from moco.builder_dist import MASA
from moco.low_rank_modules import apply_low_rank_structure
from training.study_state import trusted_load


FLOP_CONVENTION = {
    'multiply_add': '2 FLOPs per MAC; dense contractions use 2 * output elements * contraction size.',
    'bias': 'One additional addition per output element when bias is present.',
    'scalar_arithmetic': 'Add/subtract/multiply/divide/sqrt/exp/erf each cost one per element; square costs one multiplication.',
    'softmax': 'Stable softmax: subtract, exp, sum, divide; 4*N - number of rows. Max comparisons excluded.',
    'batch_norm': 'Unfolded inference expression: subtract/divide, optional affine multiply/add, plus epsilon and sqrt per channel.',
    'excluded': 'Memory movement, casts, indexing, integer shape arithmetic, comparisons, ReLU and max pooling comparisons; eval dropout is identity.',
    'precision': 'Arithmetic count independent of bit width; includes explicit weight dequantization arithmetic in the traced forward.',
    'scope': 'Fixed-input inference arithmetic estimate, not hardware instructions or training FLOPs; unknown operators are fatal.',
}


def count_forward_flops(model, batch):
    """Count the executed tensor-operation graph, including functional attention."""
    if any(module.training for module in model.modules()):
        raise ValueError('FLOPs must be counted in evaluation mode.')
    captured = []
    def record_warning(message, category, filename, lineno, file=None, line=None):
        captured.append((message, category, filename, lineno, traceback.extract_stack()))

    with warnings.catch_warnings(), torch.no_grad():
        warnings.simplefilter('always', torch.jit.TracerWarning)
        warnings.showwarning = record_warning
        graph, _ = torch.jit._get_trace_graph(model, (batch,))
    trace_warnings = []
    for message, category, filename, lineno, stack in captured:
        source = linecache.getline(filename, lineno).strip()
        trace_warnings.append(dict(message=str(message), source=source))
        if issubclass(category, torch.jit.TracerWarning):
            audited_shape_checks = ('assert A.size(0) == self.kernel_size',
                                    'shape[shape.index(-1)] = int(x.size(-1) / -np.prod(shape))',
                                    'scores = q @ k.transpose(-2, -1) / np.sqrt(k.size(-1))')
            # NumPy conversion warnings inside split_last originate in Torch;
            # identify the audited shape-only caller, not a generic Tensor conversion.
            from_shape_helper = any(Path(frame.filename).name == 'CMABert_backbone.py' and
                                    frame.line in audited_shape_checks for frame in stack)
            if source not in audited_shape_checks and not from_shape_helper:
                raise ValueError('Unaudited tracing warning; possible data-dependent FLOP path: ' + source +
                                 ' | callers: ' + repr([(Path(frame.filename).name, frame.name, frame.line)
                                                       for frame in stack[-12:]]))
    by_operator, ignored, unsupported = Counter(), Counter(), Counter()
    primitives = {'prim::Constant', 'prim::ListConstruct', 'prim::ListUnpack',
                  'prim::TupleConstruct', 'prim::TupleUnpack', 'prim::DictConstruct', 'prim::NumToTensor'}
    no_arithmetic = {'aten::size', 'aten::Int', 'aten::ScalarImplicit', 'aten::to', 'aten::type_as',
                     'aten::contiguous', 'aten::clone', 'aten::detach', 'aten::view', 'aten::reshape',
                     'aten::flatten', 'aten::permute', 'aten::transpose', 'aten::t', 'aten::cat',
                     'aten::slice', 'aten::select', 'aten::index', 'aten::index_select', 'aten::gather',
                     'aten::unsqueeze', 'aten::squeeze', 'aten::expand', 'aten::expand_as', 'aten::repeat',
                     'aten::ones', 'aten::zeros', 'aten::ones_like', 'aten::zeros_like', 'aten::full',
                     'aten::lift', 'aten::lift_fresh', 'aten::masked_fill', 'aten::masked_fill_',
                     'aten::eq', 'aten::gt', 'aten::ge', 'aten::lt', 'aten::le', 'aten::ne',
                     'aten::relu', 'aten::relu_', 'aten::max_pool1d', 'aten::max_pool2d', 'aten::max_pool3d'}

    def shape(value):
        sizes = value.type().sizes()
        if sizes is None or any(size is None for size in sizes):
            raise ValueError('Unknown tensor shape in FLOP graph: ' + str(value))
        return list(sizes)

    def numel(value):
        return math.prod(shape(value))

    def floating(value):
        kind = value.type()
        return hasattr(kind, 'scalarType') and kind.scalarType() in ('Half', 'Float', 'Double', 'BFloat16')

    for node in graph.nodes():
        op = node.kind()
        ins, outs = list(node.inputs()), list(node.outputs())
        if op in primitives or op in no_arithmetic:
            ignored[op] += 1
            continue
        try:
            if op in ('aten::dropout', 'aten::dropout_'):
                if ins[2].toIValue() is not False:
                    raise ValueError('Non-evaluation dropout in trace.')
                ignored[op] += 1
                continue
            n = numel(outs[0])
            if op in ('aten::mul', 'aten::div', 'aten::floor_divide', 'aten::add', 'aten::add_',
                      'aten::sub', 'aten::sub_', 'aten::rsub', 'aten::sqrt', 'aten::exp', 'aten::erf'):
                if not floating(outs[0]):
                    ignored[op + ':integer'] += 1
                    continue
                if op == 'aten::floor_divide':
                    raise ValueError('Floating floor division needs an explicit convention.')
                cost = n
                if op in ('aten::add', 'aten::add_', 'aten::sub', 'aten::sub_', 'aten::rsub'):
                    alpha = ins[2].toIValue() if len(ins) > 2 else 1
                    if alpha is None:
                        raise ValueError('Unknown arithmetic alpha.')
                    if alpha != 1:
                        cost += n
            elif op == 'aten::pow':
                if ins[1].toIValue() != 2:
                    raise ValueError('Only square has an audited pow formula.')
                cost = n
            elif op == 'aten::linear':
                cost = 2 * n * shape(ins[0])[-1]
                if ins[2].type().kind() != 'NoneType':
                    cost += n
            elif op in ('aten::_convolution', 'aten::conv1d', 'aten::conv2d', 'aten::conv3d'):
                if op == 'aten::_convolution' and ins[6].toIValue() is not False:
                    raise ValueError('Transposed convolution needs a separate formula.')
                weight = shape(ins[1])
                cost = 2 * n * math.prod(weight[1:])
                if ins[2].type().kind() != 'NoneType':
                    cost += n
            elif op in ('aten::matmul', 'aten::mm', 'aten::bmm'):
                cost = 2 * n * shape(ins[0])[-1]
            elif op == 'aten::einsum':
                equation = ins[0].toIValue()
                terms, result = equation.replace(' ', '').split('->')
                terms = terms.split(',')
                tensors = list(ins[1].node().inputs())
                if len(terms) != 2 or len(tensors) != 2 or '.' in equation:
                    raise ValueError('Only explicit two-operand einsum is audited.')
                dimensions = {}
                for term, tensor in zip(terms, tensors):
                    sizes = shape(tensor)
                    if len(term) != len(sizes) or len(set(term)) != len(term):
                        raise ValueError('Unsupported einsum rank/diagonal.')
                    for label, size in zip(term, sizes):
                        if label in dimensions and dimensions[label] != size:
                            raise ValueError('Broadcast einsum needs a separate formula.')
                        dimensions[label] = size
                contraction = math.prod(size for label, size in dimensions.items() if label not in result)
                cost = 2 * n * contraction
            elif op in ('aten::sum', 'aten::mean'):
                cost = numel(ins[0]) - n + (n if op == 'aten::mean' else 0)
            elif op in ('aten::softmax', 'aten::_softmax'):
                dim = ins[1].toIValue()
                if dim is None:
                    raise ValueError('Unknown softmax dimension.')
                cost = 4 * n - n // shape(ins[0])[dim]
            elif op == 'aten::batch_norm':
                if ins[5].toIValue() is not False:
                    raise ValueError('Training batch normalization in trace.')
                affine = ins[1].type().kind() != 'NoneType'
                cost = (4 if affine else 2) * n + 2 * shape(ins[0])[1]
            else:
                unsupported[op] += 1
                continue
            by_operator[op] += int(cost)
        except (ValueError, RuntimeError, AttributeError, TypeError, IndexError) as error:
            raise ValueError('Cannot safely count {}: {}'.format(op, error)) from error
    if unsupported:
        raise ValueError('Unsupported FLOP operators (no partial total exported): ' + repr(dict(unsupported)))
    return dict(flops=sum(by_operator.values()), by_operator=dict(by_operator),
                ignored_operators=dict(ignored), unsupported_operators=dict(unsupported),
                trace_warnings=trace_warnings, convention=FLOP_CONVENTION)


def preflight_flop_layouts(inputs, classes, device):
    """Reject unsupported runtime graph layouts before any training stage."""
    audits = {}
    for architecture in ('baseline', 'ghost', 'lowrank'):
        model = MASA(skeleton_representation='graph-based', num_class=classes, pretrain=False,
                     dropout=0.0, use_ghost_conv=architecture == 'ghost', ghost_ratio=2, ghost_mode='all')
        if architecture == 'lowrank':
            apply_low_rank_structure(model, 0.125, 'transformer,conv,project_head', 64)
        model = model.to(device).eval()
        for quantized in (False, True):
            if quantized:
                replace_int8_supported_layers(model)
                if device.type == 'cuda':
                    model.half()
                model.to(device).eval()
            dtype = infer_model_input_dtype(model)
            batch = {key: value.to(device=device, dtype=dtype) for key, value in inputs.items()}
            name = architecture + ('_int8_fp16' if quantized and device.type == 'cuda'
                                   else '_int8_fp32_cpu_test' if quantized else '_fp32')
            audits[name] = count_forward_flops(model, batch)
        del model, batch
        if device.type == 'cuda':
            torch.cuda.empty_cache()
    return audits


def parameter_count(model):
    # Quantization moves learned weights to qweight buffers, not Parameters.
    return sum(value.numel() for value in model.parameters()) + sum(
        module.qweight.numel() for module in model.modules() if hasattr(module, 'qweight'))


def tensor_fingerprint(state):
    digest = hashlib.sha256()
    for name, value in sorted(state.items()):
        name = name[7:] if name.startswith('module.') else name
        tensor = value.detach().cpu().contiguous()
        header = repr((name, str(tensor.dtype), tuple(tensor.shape))).encode('utf-8')
        digest.update(len(header).to_bytes(8, 'little'))
        digest.update(header)
        digest.update(tensor.numpy().tobytes())
    return digest.hexdigest()


def audit_weights(checkpoints, output):
    fingerprints, groups = {}, {}
    for stage, path in checkpoints.items():
        checkpoint = trusted_load(path)
        fingerprint = tensor_fingerprint(checkpoint.get('state_dict', checkpoint))
        fingerprints[stage] = dict(checkpoint=path, tensor_sha256=fingerprint)
        groups.setdefault(fingerprint, []).append(stage)
        del checkpoint
    duplicates = [stages for stages in groups.values() if len(stages) > 1]
    write_json(output, dict(stages=fingerprints, duplicate_groups=duplicates,
                           scope='All model-state tensors; ignores checkpoint packaging and optimizer metadata.'))
    if duplicates:
        raise ValueError('Identical model states across distinct training stages; inspect weight_audit.json: ' + repr(duplicates))


def load_model(entry, classes):
    model = MASA(skeleton_representation='graph-based', num_class=classes, pretrain=False,
                 dropout=0.0, use_ghost_conv=entry['architecture'] == 'ghost',
                 ghost_ratio=2, ghost_mode='all')
    if entry['architecture'] == 'lowrank':
        apply_low_rank_structure(model, 0.125, 'transformer,conv,project_head', 64)
    checkpoint = trusted_load(entry['checkpoint'])
    state = checkpoint.get('state_dict', checkpoint)
    state = {key[7:] if key.startswith('module.') else key: value for key, value in state.items()}
    model.load_state_dict(state, strict=True)
    del checkpoint, state
    if entry['quantized']:
        replace_int8_supported_layers(model)
        model.half()
        model.load_state_dict(trusted_load(entry['quantized_checkpoint']), strict=True)
    return model.eval()


def percentiles(values):
    ordered = sorted(values)

    def percentile(p):
        position = (len(ordered) - 1) * p
        low = int(position)
        high = min(low + 1, len(ordered) - 1)
        return ordered[low] + (ordered[high] - ordered[low]) * (position - low)

    return dict(median_ms=statistics.median(ordered), mean_ms=statistics.mean(ordered),
                p25_ms=percentile(0.25), p75_ms=percentile(0.75),
                p95_ms=percentile(0.95), min_ms=ordered[0], max_ms=ordered[-1])


def benchmark(args):
    with ExitStack() as stack:
        return _benchmark(args, stack)


def _benchmark(args, stack):
    if min(args.warmup, args.repetitions, args.rounds) < 1:
        raise ValueError('Warmup, repetitions, and rounds must be positive.')
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError('Exactly one visible CUDA GPU is required.')
    uuid = gpu_uuid()
    manifest = read_json(args.manifest)
    inherited = (os.environ.get('MASA_STUDY_LOCK_OWNER') == str(os.getppid())
                 and os.environ.get('MASA_STUDY_LOCK_GPU') == uuid)
    if not inherited:
        directory = Path(manifest['gpu_lock_directory'])
        directory.mkdir(parents=True, exist_ok=True)
        lock = stack.enter_context((directory / (uuid + '.study.lock')).open('a'))
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    assert_idle(uuid)
    # These settings apply to this subprocess only, not to training.
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    torch.manual_seed(123)
    _, data, collate = evaluation_data(manifest)
    if len(data) < manifest['batch_size']:
        raise ValueError('Not enough test clips for a full fixed-size benchmark batch.')
    indexes = list(range(manifest['batch_size']))
    inputs, _ = collate([data[index] for index in indexes])
    args.out.parent.mkdir(parents=True, exist_ok=True)
    audit_weights(manifest['training_checkpoints'], args.out.parent / 'weight_audit.json')
    input_file = args.out.parent / 'benchmark_inputs.pth'
    torch.save(dict(indexes=indexes, inputs=inputs), input_file)
    order = list(manifest['models'])
    random.Random(123).shuffle(order)
    records = {}
    for entry in order:
        assert_idle(uuid)
        print('[BENCHMARK] ' + entry['id'], flush=True)
        model = load_model(entry, manifest['classes'])
        tensor_count = state_dict_numel(model)
        logical_parameters = parameter_count(model)
        registered_parameters = sum(parameter.numel() for parameter in model.parameters())
        # Use one canonical filename/serializer for every model size measurement.
        snapshot = args.out.parent / 'benchmark_model_state.pth'
        torch.save(model.state_dict(), snapshot)
        size_mb = snapshot.stat().st_size / (1024 * 1024)
        snapshot.unlink()
        model = model.cuda()
        dtype = infer_model_input_dtype(model)
        batch = {key: value.to(device='cuda', dtype=dtype) for key, value in inputs.items()}
        measurements, round_results = [], []
        with torch.no_grad():
            logits = model(batch)
            if not torch.isfinite(logits).all():
                raise ValueError('Nonfinite benchmark logits: ' + entry['id'])
            for _ in range(args.rounds):
                assert_idle(uuid)
                for _ in range(args.warmup):
                    model(batch)
                torch.cuda.synchronize()
                series = []
                for _ in range(args.repetitions):
                    torch.cuda.synchronize()
                    started = time.perf_counter()
                    model(batch)
                    torch.cuda.synchronize()
                    series.append((time.perf_counter() - started) * 1000.0)
                measurements.extend(series)
                round_results.append(percentiles(series))
                assert_idle(uuid)
            # Trace separately after timings; no instrumentation affects latency.
            flop_audit = count_forward_flops(model, batch)
            torch.cuda.synchronize()
            flops = flop_audit['flops']
        record = percentiles(measurements)
        record.update(latency_ms_per_batch=record['median_ms'], batch_size=manifest['batch_size'],
                      latency_ms_per_sample_equivalent=record['median_ms'] / manifest['batch_size'],
                      flops_per_batch=flops, param_count_tensors=tensor_count,
                      gflops_per_sample=flops / manifest['batch_size'] / 1e9, flop_audit=flop_audit,
                      parameter_count=logical_parameters, parameter_numel=registered_parameters, model_size_mb=size_mb,
                      precision='INT8 stored weights with FP16 fallback/arithmetic' if entry['quantized'] else 'FP32',
                      checkpoint_sha256=sha256(entry.get('quantized_checkpoint', entry['checkpoint'])),
                      rounds=round_results, measurements_ms=measurements)
        records[entry['id']] = record
        del model, batch, logits
        gc.collect()
        torch.cuda.empty_cache()
    assert_idle(uuid)
    write_json(args.out, dict(dataset=manifest['dataset'], gpu=torch.cuda.get_device_name(0),
                             gpu_uuid=uuid, torch=torch.__version__, cuda=torch.version.cuda,
                             warmup_per_round=args.warmup, repetitions_per_round=args.repetitions,
                             rounds=args.rounds, tf32=False, cudnn_benchmark=False,
                             timing='GPU-resident forward only; synchronized wall clock; no FLOP hooks or data loading.',
                             flop_scope='Traced inference arithmetic including attention, GCN contractions, elementwise arithmetic and explicit dequantization. Normalized per sample from batch 32.',
                             flop_convention=FLOP_CONVENTION,
                             param_count_scope='Legacy-compatible state-dict tensor elements, including buffers/scales.',
                             parameter_count_scope='Learned weight/bias elements, including INT8 qweight buffers; excludes scales and running statistics.',
                             input_shapes={key: list(value.shape) for key, value in inputs.items()},
                             input_sha256=sha256(input_file), model_order=[entry['id'] for entry in order], models=records))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--warmup', type=int, default=30)
    parser.add_argument('--repetitions', type=int, default=100)
    parser.add_argument('--rounds', type=int, default=3)
    benchmark(parser.parse_args())
