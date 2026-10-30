#!/usr/bin/env python3
"""Sequential low-rank KD extension of all six completed compression studies."""

import argparse
import copy
import csv
import fcntl
import importlib.util
import json
import os
import sys
import time
from contextlib import ExitStack
from dataclasses import asdict
from pathlib import Path
from types import SimpleNamespace

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from experiments.run_study import (
    LOW_TARGETS, TRAIN, Task, assert_idle, build_plan, evaluation_data, execute,
    export, gpu_uuid, read_json, remove_measurement_copies, signatures,
    source_record, validate_outputs, write_json,
)


DATASETS = ('wlasl100', 'wlasl300', 'wlasl2000', 'msasl100', 'msasl200', 'msasl1000')


def build_extension_plan(repo, python, workspace, study_root, output_root,
                         dataset, workers=4, original=None):
    if dataset not in DATASETS:
        raise ValueError('Unsupported dataset: ' + dataset)
    family = 'wlasl' if dataset.startswith('wlasl') else 'msasl'
    classes = int(dataset[len(family):])
    old = Path(study_root) / dataset
    output = Path(output_root) / dataset
    _, expected = build_plan(repo, python, workspace, family, classes, old, workers)
    original = copy.deepcopy(expected if original is None else original)
    if any(original[key] != expected[key] for key in ('dataset', 'family', 'classes',
                                                      'target_t', 'batch_size', 'temporal_sampling')):
        raise ValueError('Original dataset/protocol does not match: ' + dataset)
    if (len(original['models']) != 14 or
            {m['id'] for m in original['models']} != {m['id'] for m in expected['models']}):
        raise ValueError('Expected the original 14-model study: ' + dataset)
    if set(original['training_checkpoints']) != set(expected['training_checkpoints']):
        raise ValueError('Expected all nine original training checkpoints: ' + dataset)
    models = original['models']
    by_id = {m['id']: m for m in models}
    expected_models = {m['id']: m for m in expected['models']}
    for entry in models:
        for key in ('architecture', 'quantized'):
            if entry[key] != expected_models[entry['id']][key]:
                raise ValueError('Original architecture/precision mismatch: ' + entry['id'])
    for model_id, stage in (('baseline_fp32', 'baseline_train'),
                            ('m1_lowrank_fp32', 'm1_lowrank_train'),
                            ('m2_lowrank_fp32', 'm2_lowrank_stage2_train')):
        if Path(by_id[model_id]['checkpoint']).resolve() != Path(original['training_checkpoints'][stage]).resolve():
            raise ValueError('Original model does not point to its intended training stage: ' + model_id)
    teacher = by_id['baseline_fp32']['checkpoint']
    if by_id['baseline_fp32']['architecture'] != 'baseline':
        raise ValueError('The KD teacher must be the dense baseline.')
    common = ['--data-root', original['data_root'], '--num-class', str(classes)]
    if family == 'wlasl':
        common += ['--subset-num', str(classes)]
    evaluate = common + ['--target-t', '32', '--batch-size', '32', '--workers', str(workers),
                         '--dropout', '0.0', '--warmup-steps', '5', '--temporal-sampling', 'index']
    low_rank_eval = ['--use-low-rank', '--low-rank-ratio', '0.125',
                     '--low-rank-targets', LOW_TARGETS, '--low-rank-min-features', '64']
    tasks = []
    initializers = {}
    for method in ('m1', 'm2'):
        initial = by_id[method + '_lowrank_fp32']
        if initial['architecture'] != 'lowrank' or initial['quantized']:
            raise ValueError('KD must start from an FP32 low-rank checkpoint.')
        initializers[method] = initial['checkpoint']
        directory = output / method / 'lowrank_all_r0125'
        fp32 = directory / 'kd_fp32'
        checkpoint = fp32 / 'best.pth.tar'
        command = [str(python), str(Path(repo) / 'training' / family / 'finetune_ghost_kd.py')]
        command += common + TRAIN + ['--workers', str(workers), '--dropout', '0.0',
                    '--teacher-ckpt', teacher, '--student-ckpt', initial['checkpoint'],
                    '--kd-alpha', '0.5', '--kd-temp', '4.0', '--student-use-low-rank',
                    '--student-low-rank-ratio', '0.125', '--student-low-rank-targets', LOW_TARGETS,
                    '--student-low-rank-min-features', '64', '--out-dir', str(fp32)]
        train = Task(method + '_lowrank_kd_train', 'train', command,
                     [str(checkpoint), str(fp32 / 'summary.json'), str(fp32 / 'metrics.json'),
                      str(fp32 / 'last.pth.tar')], [teacher, initial['checkpoint']])
        tasks.append(train)
        original['training_checkpoints'][train.name] = str(checkpoint)
        report = fp32 / 'evaluation.json'
        predictions = fp32 / 'evaluation_state_dict.predictions.json'
        evaluator = 'checkpoint_metrics.py' if family == 'wlasl' else 'msasl_checkpoint_metrics.py'
        tasks.append(Task(method + '_lowrank_kd_fp32_eval', 'evaluate',
                          [str(python), str(Path(repo) / 'analysis' / evaluator)] + evaluate +
                          ['--ckpt', str(checkpoint), '--out', str(report)] + low_rank_eval,
                          [str(report), str(predictions)], [str(checkpoint)]))
        models.append(dict(id=method + '_lowrank_kd_fp32', architecture='lowrank', quantized=False,
                           checkpoint=str(checkpoint), report=str(report), predictions=str(predictions)))
        quantized = directory / 'kd_quantized'
        report = quantized / 'summary.json'
        snapshot = quantized / 'after_int8_fp16_state_dict.pth'
        predictions = quantized / 'after_int8_fp16_state_dict.predictions.json'
        quantizer = 'quantization_report.py' if family == 'wlasl' else 'msasl_quantization_report.py'
        tasks.append(Task(method + '_lowrank_kd_quantized', 'quantize',
                          [str(python), str(Path(repo) / 'analysis' / quantizer)] + evaluate +
                          ['--finetuned-ckpt', str(checkpoint), '--out-dir', str(quantized)] + low_rank_eval,
                          [str(report), str(snapshot), str(predictions)], [str(checkpoint)]))
        models.append(dict(id=method + '_lowrank_kd_quantized', architecture='lowrank', quantized=True,
                           checkpoint=str(checkpoint), quantized_checkpoint=str(snapshot),
                           report=str(report), predictions=str(predictions)))
    original['extension'] = dict(teacher=teacher, student_initializers=initializers,
                                 original_manifest=str(old / 'models.json'),
                                 note='M1 original best; M2 original stage-2 best; exact factors, no SVD.')
    original['gpu_lock_directory'] = str(Path(workspace) / 'logs')
    manifest_path = output / 'models.json'
    benchmark = output / 'comparisons/standardized_benchmark.json'
    # Remeasure the old models too: all 18 variants share this GPU and input batch.
    tasks.append(Task('standardized_benchmark', 'benchmark',
                      [str(python), str(Path(repo) / 'analysis/benchmark_study.py'),
                       '--manifest', str(manifest_path), '--warmup', '30', '--repetitions', '100',
                       '--rounds', '3', '--out', str(benchmark)],
                      [str(benchmark), str(benchmark.parent / 'benchmark_inputs.pth'),
                       str(benchmark.parent / 'weight_audit.json')],
                      list(dict.fromkeys([m['checkpoint'] for m in models] +
                                         [m['quantized_checkpoint'] for m in models if m['quantized']] +
                                         list(original['training_checkpoints'].values())))))
    tasks.append(Task('export_comparisons', 'export',
                      [str(python), str(Path(repo) / 'experiments/run_lowrank_kd_study.py'),
                       '--export-manifest', str(manifest_path), '--benchmark', str(benchmark),
                       '--out-dir', str(benchmark.parent)],
                      [str(benchmark.parent / 'results.json'), str(benchmark.parent / 'results.csv')],
                      [str(benchmark)] + [p for m in models for p in (m['report'], m['predictions'])]))
    assert len(tasks) == 8 and len(models) == 18
    return tasks, original


def import_kd_trainer(family):
    directory = REPO / 'training' / family
    previous = sys.modules.pop('finetune', None)
    name = family + '_extension_kd'
    sys.path.insert(0, str(directory))
    try:
        spec = importlib.util.spec_from_file_location(name, directory / 'finetune_ghost_kd.py')
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        spec.loader.exec_module(module)
        return module
    finally:
        sys.path.pop(0)
        sys.modules.pop('finetune', None)
        if previous is not None:
            sys.modules['finetune'] = previous


def check_initialization(manifest):
    """Disposable real-data KD backward, using each actual factorized student."""
    import gc
    import torch
    module = import_kd_trainer(manifest['family'])
    module.set_seed(123)
    assert_idle(gpu_uuid())
    _, dataset, collate = evaluation_data(manifest)
    inputs, labels = collate([dataset[0], dataset[1]])
    inputs = {key: value.cuda() for key, value in inputs.items()}
    labels = labels.cuda()
    teacher = module.build_model(manifest['classes'], 0.0, False, 2, 'all')
    teacher_info = module.load_exact_checkpoint(teacher, manifest['extension']['teacher'])
    teacher = teacher.cuda().eval()
    for parameter in teacher.parameters():
        parameter.requires_grad = False
    details = {}
    for method, checkpoint in manifest['extension']['student_initializers'].items():
        args = SimpleNamespace(num_class=manifest['classes'], dropout=0.0,
                               student_use_ghost_conv=False, teacher_use_ghost_conv=False,
                               student_low_rank_ratio=0.125, student_low_rank_targets=LOW_TARGETS,
                               student_low_rank_min_features=64, student_ckpt=checkpoint)
        student, load_info, metadata = module.build_low_rank_student(args)
        student = student.cuda().train()
        with torch.no_grad():
            teacher_logits = teacher(inputs)
        logits = student(inputs)
        if logits.shape != (2, manifest['classes']) or not torch.isfinite(logits).all():
            raise ValueError('Invalid low-rank student logits: ' + method)
        loss = 0.5 * torch.nn.functional.cross_entropy(logits, labels)
        loss = loss + 0.5 * module.kd_loss_fn(logits, teacher_logits, 4.0)
        if not torch.isfinite(loss):
            raise ValueError('Nonfinite KD smoke-test loss.')
        loss.backward()
        gradients = [p.grad for p in student.parameters() if p.grad is not None]
        if not gradients or any(not torch.isfinite(g).all() for g in gradients):
            raise ValueError('Missing/nonfinite low-rank student gradients.')
        torch.cuda.synchronize()
        details[method] = dict(checkpoint=checkpoint, load_info=load_info, configuration=metadata,
                               smoke_loss=loss.item(), gradients_checked=len(gradients))
        del student, logits, loss, gradients, teacher_logits
        gc.collect()
        torch.cuda.empty_cache()
    print('[PASS] Exact low-rank restoration and real-data KD backward for M1/M2.', flush=True)
    return dict(teacher_load=teacher_info, students=details,
                scope='Disposable batch of two; no optimizer step or checkpoint modification.')


def check_roots(study_root, output_root):
    source, output = Path(study_root).resolve(), Path(output_root).resolve()
    common = Path(os.path.commonpath([str(source), str(output)]))
    if common in (source, output):
        raise ValueError('Extension outputs and original studies must be separate, non-nested roots.')


def original_inputs(study_root, dataset, manifest):
    directory = Path(study_root) / dataset
    paths = [directory / name for name in ('models.json', 'preflight.json',
                                          'comparisons/results.json', 'comparisons/results.csv')]
    for entry in manifest['models'][:14]:
        paths.extend(Path(entry[key]) for key in ('checkpoint', 'report', 'predictions'))
        if entry['quantized']:
            paths.append(Path(entry['quantized_checkpoint']))
    paths.extend(Path(p) for stage, p in manifest['training_checkpoints'].items()
                 if stage not in ('m1_lowrank_kd_train', 'm2_lowrank_kd_train'))
    for path in set(paths):
        if not path.is_file() or not path.stat().st_size:
            raise FileNotFoundError('Missing original study artifact: ' + str(path))
    results = read_json(directory / 'comparisons/results.json')
    if results['dataset'] != dataset or len(results['results']) != 14:
        raise ValueError('Original study is not a completed 14-model export: ' + dataset)
    # Pin initialization and protocol contents before any training begins.
    return [str(p) for p in paths[:4]] + [manifest['extension']['teacher']] + list(
        manifest['extension']['student_initializers'].values())


def run_task(task, output, resume, env):
    marker = output / 'task_records' / (task.name + '.json')
    inputs = signatures(task.sources)
    if marker.exists():
        saved = read_json(marker)
        if (saved['command'] != task.command or saved['inputs'] != inputs or
                saved['outputs'] != signatures(task.outputs)):
            raise ValueError('Completed task artifacts/configuration changed: ' + task.name)
        print('[SKIP verified] ' + task.name, flush=True)
        return
    command = list(task.command)
    if task.kind == 'train':
        directory = Path(task.outputs[0]).parent
        launch = directory / 'launch.json'
        record = dict(command=task.command, inputs=inputs)
        if launch.exists() and read_json(launch) != record:
            raise ValueError('Interrupted KD stage inputs changed: ' + task.name)
        write_json(launch, record)
        if (directory / 'last.pth.tar').exists():
            if not resume:
                raise ValueError('Interrupted training requires --resume.')
            command.append('--resume-stage')
    started = time.time()
    execute(command, output / 'logs' / (task.name + '.log'), env)
    validate_outputs(task)
    remove_measurement_copies(task)
    write_json(marker, dict(command=task.command, executed_command=command, inputs=inputs,
                            outputs=signatures(task.outputs), elapsed_seconds=time.time() - started))


def combine_results(output_root, datasets):
    rows, fields = [], None
    for dataset in datasets:
        path = output_root / dataset / 'comparisons/results.csv'
        with path.open(newline='') as handle:
            reader = csv.DictReader(handle)
            if fields is not None and fields != reader.fieldnames:
                raise ValueError('Dataset CSV schemas differ.')
            fields = reader.fieldnames
            current = list(reader)
        if len(current) != 18 or len({r['experiment'] for r in current}) != 18:
            raise ValueError('Expected 18 unique final results: ' + dataset)
        rows.extend(dict(dataset=dataset, **row) for row in current)
    temporary = output_root / 'all_results.csv.tmp'
    with temporary.open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=['dataset'] + fields)
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temporary, output_root / 'all_results.csv')


def run(args):
    check_roots(args.study_root, args.output_root)
    if args.workers < 0 or not args.datasets or len(set(args.datasets)) != len(args.datasets):
        raise ValueError('Use nonnegative workers and unique dataset names.')
    assigned = os.environ.get('SLURM_CPUS_PER_TASK')
    if assigned and args.workers > int(assigned):
        raise ValueError('Workers exceed the Slurm CPU allocation.')
    plans = []
    for dataset in args.datasets:
        old_manifest = args.study_root / dataset / 'models.json'
        original = read_json(old_manifest) if old_manifest.is_file() else None
        tasks, manifest = build_extension_plan(REPO, sys.executable, args.workspace, args.study_root,
                                               args.output_root, dataset, args.workers, original)
        plans.append((dataset, tasks, manifest))
    plan = dict(datasets=args.datasets, workers=args.workers, study_root=str(args.study_root),
                output_root=str(args.output_root),
                studies=[dict(dataset=d, tasks=[asdict(t) for t in ts], manifest=m) for d, ts, m in plans])
    if args.dry_run:
        print(json.dumps(plan, indent=2))
        return
    # Fail for any missing dataset/checkpoint before spending time on the first KD run.
    required = {d: original_inputs(args.study_root, d, m) for d, _, m in plans}
    if args.output_root.exists() and any(args.output_root.iterdir()) and not args.resume:
        raise ValueError('Output exists. Use --resume, or choose a fresh --output-root.')
    args.output_root.mkdir(parents=True, exist_ok=True)
    with ExitStack() as stack:
        lock = stack.enter_context((args.output_root / '.run.lock').open('a'))
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        uuid = gpu_uuid()
        gpu_lock = args.workspace / 'logs' / (uuid + '.study.lock')
        gpu_lock.parent.mkdir(parents=True, exist_ok=True)
        lock = stack.enter_context(gpu_lock.open('a'))
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        assert_idle(uuid)
        records = dict(plan=plan, source=source_record(REPO),
                       original_inputs={d: signatures(paths) for d, paths in required.items()})
        configuration = args.output_root / 'extension.json'
        if configuration.exists():
            saved = read_json(configuration)
            for key in ('plan', 'original_inputs'):
                if saved[key] != records[key]:
                    raise ValueError('Extension configuration or original inputs changed: ' + key)
            if saved['source']['source_sha256'] != records['source']['source_sha256']:
                raise ValueError('Source changed since launch; restore it before --resume.')
        else:
            write_json(configuration, records)
        env = dict(os.environ, PYTHONPATH=str(REPO) + os.pathsep + os.environ.get('PYTHONPATH', ''),
                   PYTHONUNBUFFERED='1', MASA_STUDY_PREDICTIONS='1',
                   MASA_STUDY_LOCK_OWNER=str(os.getpid()), MASA_STUDY_LOCK_GPU=uuid,
                   OMP_NUM_THREADS='1', MKL_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1')
        for dataset, tasks, manifest in plans:
            output = args.output_root / dataset
            write_json(output / 'models.json', manifest)
            print('[PREFLIGHT] ' + dataset + ': data, environment, FLOPs, exact low-rank checkpoints', flush=True)
            execute([sys.executable, str(REPO / 'experiments/run_study.py'), '--preflight-manifest',
                     str(output / 'models.json'), '--out', str(output / 'preflight.current.json')],
                    output / 'logs/preflight.log', env)
            current = read_json(output / 'preflight.current.json')
            original = read_json(args.study_root / dataset / 'preflight.json')
            stable = ('splits', 'metadata_sha256', 'pretrained_sha256', 'packages', 'cuda', 'python',
                      'cudnn', 'training_backend_defaults')
            for key in stable:
                if current[key] != original[key]:
                    raise ValueError('Original study data/environment differs: {} {}'.format(dataset, key))
            saved_path = output / 'preflight.json'
            if saved_path.exists():
                saved = read_json(saved_path)
                for key in stable + ('gpu', 'driver'):
                    if current[key] != saved[key]:
                        raise ValueError('Extension preflight changed on resume: ' + key)
            else:
                write_json(saved_path, current)
            execute([sys.executable, str(Path(__file__).resolve()), '--check-initialization',
                     str(output / 'models.json'), '--out', str(output / 'initialization.json')],
                    output / 'logs/initialization.log', env)
            for index, task in enumerate(tasks, 1):
                print('[{}/{} {}] {} | log: {}'.format(index, len(tasks), dataset, task.name,
                      output / 'logs' / (task.name + '.log')), flush=True)
                run_task(task, output, args.resume, env)
                print('[DONE] {} {}'.format(dataset, task.name), flush=True)
            print('[COMPLETE] ' + str(output / 'comparisons/results.csv'), flush=True)
        combine_results(args.output_root, args.datasets)
        print('[ALL COMPLETE] ' + str(args.output_root / 'all_results.csv'), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--workspace', type=Path,
                        default=Path(os.environ.get('STUDY_WORKSPACE', '/shared/g00107537/jupyter')))
    parser.add_argument('--study-root', type=Path, help='Read-only completed 14-model studies.')
    parser.add_argument('--output-root', type=Path, help='Separate directory for the KD extension.')
    parser.add_argument('--datasets', nargs='+', choices=DATASETS, default=list(DATASETS))
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--dry-run', action='store_true')
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument('--export-manifest', type=Path, help=argparse.SUPPRESS)
    modes.add_argument('--check-initialization', type=Path, help=argparse.SUPPRESS)
    parser.add_argument('--benchmark', type=Path, help=argparse.SUPPRESS)
    parser.add_argument('--out-dir', type=Path, help=argparse.SUPPRESS)
    parser.add_argument('--out', type=Path, help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.export_manifest:
        export(args.export_manifest, args.benchmark, args.out_dir, expected_count=18)
    elif args.check_initialization:
        write_json(args.out, check_initialization(read_json(args.check_initialization)))
    else:
        args.workspace = args.workspace.resolve()
        args.study_root = (args.study_root or args.workspace / 'results/compression_study_v1').resolve()
        args.output_root = (args.output_root or args.workspace / 'results/compression_study_lowrank_kd_v1').resolve()
        run(args)


if __name__ == '__main__':
    main()
