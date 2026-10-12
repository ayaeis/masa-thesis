#!/usr/bin/env python3
"""One dataset study: protocol, preflight, execution, provenance, and export."""

import argparse
import csv
import fcntl
import hashlib
import json
import math
import os
import shlex
import signal
import subprocess
import sys
import time
from contextlib import ExitStack
from dataclasses import asdict, dataclass
from pathlib import Path


# Records and hashes

def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    os.replace(temporary, path)


def read_json(path):
    return json.loads(Path(path).read_text())


# Experiment definitions

TRAIN = [
    '--epochs', '60', '--batch-size', '64', '--optim', 'sgd',
    '--momentum', '0.9', '--lr', '0.01', '--weight-decay', '0.0',
    '--scheduler', 'multistep', '--milestones', '20', '40', '--lr-gamma', '0.1',
    '--target-t', '32', '--temporal-sampling', 'index', '--freeze-epochs', '0',
    '--head-lr-mult', '1.0', '--warmup-epochs', '0', '--train-temporal-crop-min', '1.0',
    '--patience', '999', '--min-delta', '0.0', '--seed', '123', '--save-last',
]


LOW_TARGETS = 'transformer,conv,project_head'


@dataclass
class Task:
    name: str
    kind: str
    command: list
    outputs: list
    sources: list


def dataset_root(workspace, family, classes):
    base = Path(workspace) / 'datasets'
    if family == 'wlasl':
        return base / 'WLASL/latest_full_wlasl' / ('official_wlasl' + str(classes)) / 'masa_ready/WLASL'
    return base / 'MSASL/native_masa_ready' / ('MSASL1000' if classes == 1000 else 'MSASL')


def build_plan(repo, python, workspace, family, classes, output, workers=8,
               warmup=30, repetitions=100, rounds=3):
    allowed = {'wlasl': (100, 300), 'msasl': (100, 200, 1000)}
    if family not in allowed or classes not in allowed[family]:
        raise ValueError('Unsupported dataset variant: {}{}'.format(family, classes))
    repo, workspace, output = Path(repo), Path(workspace), Path(output)
    data = dataset_root(workspace, family, classes)
    pretrain = workspace / 'checkpoints/pretrained_model.pth.tar'
    common = ['--data-root', str(data), '--num-class', str(classes)]
    if family == 'wlasl':
        common += ['--subset-num', str(classes)]
    train = common + TRAIN + ['--workers', str(workers)]
    evaluate = common + ['--target-t', '32', '--batch-size', '32', '--workers', str(workers),
                         '--dropout', '0.0', '--warmup-steps', '5', '--temporal-sampling', 'index']
    tasks, models = [], []
    trainer = repo / 'training' / family
    evaluator = repo / ('analysis/checkpoint_metrics.py' if family == 'wlasl' else 'analysis/msasl_checkpoint_metrics.py')
    quantizer = repo / ('analysis/quantization_report.py' if family == 'wlasl' else 'analysis/msasl_quantization_report.py')

    def architecture(kind, training=False):
        if kind == 'ghost':
            return ['--use-ghost-conv', '--ghost-ratio', '2', '--ghost-mode', 'all']
        if kind == 'lowrank':
            return (['--rank-ratio', '0.125'] if training else ['--use-low-rank', '--low-rank-ratio', '0.125']) + [
                '--low-rank-targets', LOW_TARGETS, '--low-rank-min-features', '64']
        return []

    def training(name, directory, script, init_flags, sources, flags=()):
        directory = output / directory
        checkpoint = directory / 'best.pth.tar'
        tasks.append(Task(name, 'train', [str(python), str(trainer / script)] + train + list(init_flags)
                          + list(flags) + ['--out-dir', str(directory)],
                          [str(checkpoint), str(directory / 'summary.json'), str(directory / 'metrics.json'),
                           str(directory / 'last.pth.tar')], [str(p) for p in sources]))
        return checkpoint

    def results(name, checkpoint, directory, kind):
        directory = output / directory
        report = directory / 'evaluation.json'
        tasks.append(Task(name + '_eval', 'evaluate', [str(python), str(evaluator)] + evaluate + [
            '--ckpt', str(checkpoint), '--out', str(report)] + architecture(kind),
            [str(report)], [str(checkpoint)]))
        models.append(dict(id=name, architecture=kind, quantized=False,
                           checkpoint=str(checkpoint), report=str(report),
                           predictions=str(directory / 'evaluation_state_dict.predictions.json')))
        tasks[-1].outputs.append(models[-1]['predictions'])

    def quantized(name, checkpoint, directory, kind):
        directory = output / directory
        report = directory / 'summary.json'
        snapshot = directory / 'after_int8_fp16_state_dict.pth'
        tasks.append(Task(name, 'quantize', [str(python), str(quantizer)] + evaluate + [
            '--finetuned-ckpt', str(checkpoint), '--out-dir', str(directory)] + architecture(kind),
            [str(report), str(snapshot)], [str(checkpoint)]))
        models.append(dict(id=name, architecture=kind, quantized=True, checkpoint=str(checkpoint),
                           quantized_checkpoint=str(snapshot), report=str(report),
                           predictions=str(directory / 'after_int8_fp16_state_dict.predictions.json')))
        tasks[-1].outputs.append(models[-1]['predictions'])

    def kd(name, checkpoint, directory):
        return training(name, directory, 'finetune_ghost_kd.py',
                        ['--teacher-ckpt', str(baseline), '--student-ckpt', str(checkpoint)],
                        [baseline, checkpoint], ['--dropout', '0.0', '--kd-alpha', '0.5', '--kd-temp', '4.0',
                        '--student-use-ghost-conv', '--student-ghost-ratio', '2', '--student-ghost-mode', 'all'])

    baseline = training('baseline_train', 'baseline/fp32', 'finetune.py',
                        ['--pretrained', str(pretrain)], [pretrain])
    results('baseline_fp32', baseline, 'baseline/fp32', 'baseline')
    quantized('baseline_quantized', baseline, 'baseline/quantized', 'baseline')

    first = 'method1_post_training'
    ghost = training('m1_ghost_train', first + '/ghost_allk/fp32', 'finetune.py',
                     ['--pretrained', str(baseline)], [baseline], architecture('ghost'))
    results('m1_ghost_fp32', ghost, first + '/ghost_allk/fp32', 'ghost')
    quantized('m1_ghost_quantized', ghost, first + '/ghost_allk/quantized', 'ghost')
    distilled = kd('m1_ghost_kd_train', ghost, first + '/ghost_allk/kd_fp32')
    results('m1_ghost_kd_fp32', distilled, first + '/ghost_allk/kd_fp32', 'ghost')
    quantized('m1_ghost_kd_quantized', distilled, first + '/ghost_allk/kd_quantized', 'ghost')
    lowrank = training('m1_lowrank_train', first + '/lowrank_all_r0125/fp32', 'finetune_lowrank.py',
                       ['--dense-ckpt', str(baseline)], [baseline], architecture('lowrank', True))
    results('m1_lowrank_fp32', lowrank, first + '/lowrank_all_r0125/fp32', 'lowrank')
    quantized('m1_lowrank_quantized', lowrank, first + '/lowrank_all_r0125/quantized', 'lowrank')

    second = 'method2_architecture_first'
    g1 = training('m2_ghost_stage1_train', second + '/ghost_allk/stage1', 'finetune.py',
                  ['--pretrained', str(pretrain)], [pretrain], architecture('ghost'))
    g2 = training('m2_ghost_stage2_train', second + '/ghost_allk/stage2', 'finetune.py',
                  ['--pretrained', str(g1)], [g1], architecture('ghost'))
    results('m2_ghost_fp32', g2, second + '/ghost_allk/stage2', 'ghost')
    quantized('m2_ghost_quantized', g2, second + '/ghost_allk/quantized', 'ghost')
    distilled = kd('m2_ghost_kd_train', g2, second + '/ghost_allk/kd_fp32')
    results('m2_ghost_kd_fp32', distilled, second + '/ghost_allk/kd_fp32', 'ghost')
    quantized('m2_ghost_kd_quantized', distilled, second + '/ghost_allk/kd_quantized', 'ghost')
    l1 = training('m2_lowrank_stage1_train', second + '/lowrank_all_r0125/stage1',
                  'finetune_lowrank_architecture_first.py',
                  ['--init-ckpt', str(pretrain), '--init-mode', 'pretrained_dense'], [pretrain], architecture('lowrank', True))
    l2 = training('m2_lowrank_stage2_train', second + '/lowrank_all_r0125/stage2',
                  'finetune_lowrank_architecture_first.py',
                  ['--init-ckpt', str(l1), '--init-mode', 'resume_low_rank'], [l1], architecture('lowrank', True))
    results('m2_lowrank_fp32', l2, second + '/lowrank_all_r0125/stage2', 'lowrank')
    quantized('m2_lowrank_quantized', l2, second + '/lowrank_all_r0125/quantized', 'lowrank')

    manifest = output / 'models.json'
    benchmark = output / 'comparisons/standardized_benchmark.json'
    tasks.append(Task('standardized_benchmark', 'benchmark', [str(python), str(repo / 'analysis/benchmark_study.py'),
                      '--manifest', str(manifest), '--warmup', str(warmup), '--repetitions', str(repetitions),
                      '--rounds', str(rounds), '--out', str(benchmark)],
                      [str(benchmark), str(output / 'comparisons/benchmark_inputs.pth'),
                       str(output / 'comparisons/weight_audit.json')],
                      [m.get('quantized_checkpoint', m['checkpoint']) for m in models]))
    tasks.append(Task('export_comparisons', 'export', [str(python), str(repo / 'experiments/run_study.py'),
                      '--export-manifest', str(manifest), '--benchmark', str(benchmark),
                      '--out-dir', str(output / 'comparisons')],
                      [str(output / 'comparisons/results.json'), str(output / 'comparisons/results.csv')],
                      [str(benchmark)] + [p for m in models for p in (m['report'], m['predictions'])]))
    metadata = dict(dataset=family + str(classes), family=family, classes=classes,
                    data_root=str(data), target_t=32, batch_size=32, temporal_sampling='index',
                    pretrained_checkpoint=str(pretrain), gpu_lock_directory=str(workspace / 'logs'), models=models,
                    training_checkpoints={task.name: task.outputs[0] for task in tasks if task.kind == 'train'})
    assert len(models) == 14
    return tasks, metadata


# GPU ownership checks

def gpu_uuid():
    rows = subprocess.check_output(
        ['nvidia-smi', '--query-gpu=uuid', '--format=csv,noheader'], text=True
    ).strip().splitlines()
    visible = os.environ.get('CUDA_VISIBLE_DEVICES', '').split(',')[0].strip()
    if len(rows) == 1:
        return rows[0].strip()
    if visible.startswith('GPU-'):
        matches = [row.strip() for row in rows if row.strip().startswith(visible)]
        if len(matches) == 1:
            return matches[0]
    # UUID selection avoids ambiguity between CUDA and nvidia-smi index ordering.
    raise RuntimeError('On a multi-GPU node, set CUDA_VISIBLE_DEVICES to one GPU UUID before launch.')


def assert_idle(uuid):
    output = subprocess.check_output(
        ['nvidia-smi', '--query-compute-apps=pid,gpu_uuid', '--format=csv,noheader,nounits'], text=True
    )
    busy = []
    for line in output.splitlines():
        columns = [part.strip() for part in line.split(',')]
        if len(columns) == 2 and columns[1] == uuid and columns[0] != str(os.getpid()):
            busy.append(columns[0])
    if busy:
        raise RuntimeError('Selected GPU has other compute processes: ' + ', '.join(busy))


# Dataset and model preflight

def evaluation_data(manifest):
    if manifest['family'] == 'wlasl':
        from feeder.single_dataset.WLASL import WLASL
        from analysis.quantization_report import WLASLSupervisedEval, collate_supervised
        base = WLASL(manifest['data_root'], data_split='test',
                     subset_num=manifest['classes'], use_cache=False)
        return base, WLASLSupervisedEval(base, manifest['target_t'], manifest['temporal_sampling']), collate_supervised
    from data_prep.msasl.archive_loader import load_archive
    from analysis.msasl_quantization_report import MSASLSupervisedEval, collate_supervised
    base = load_archive(manifest['data_root'], data_split='test',
                        class_num=manifest['classes'], use_cache=False)
    return base, MSASLSupervisedEval(base, manifest['target_t'], manifest['temporal_sampling']), collate_supervised


def preflight(manifest):
    import importlib.metadata
    import torch

    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError('The study requires exactly one visible CUDA GPU per dataset launch.')
    uuid = gpu_uuid()
    assert_idle(uuid)
    if manifest['family'] == 'msasl':
        unavailable = 'unavailable_records.json' if manifest['classes'] == 1000 else 'unavailable_msasl200_records.json'
        for name in (unavailable, 'native_feeder_exclusions.json'):
            if not (Path(manifest['data_root']) / name).is_file():
                raise FileNotFoundError('Required archive exclusion metadata is missing: ' + name)
    if manifest['family'] == 'wlasl':
        from feeder.single_dataset.WLASL import WLASL
        train = WLASL(manifest['data_root'], data_split='train', subset_num=manifest['classes'], use_cache=False)
    else:
        from data_prep.msasl.archive_loader import load_archive
        train = load_archive(manifest['data_root'], data_split='train', class_num=manifest['classes'], use_cache=False)
    test, eval_dataset, collate = evaluation_data(manifest)
    splits = {'train': train.video_list[:train.flag], 'val': train.video_list[train.flag:],
              'test': test.video_list}
    root = Path(manifest['data_root'])
    details, file_sets = {}, {}
    for split, rows in splits.items():
        if len(rows) == 0:
            raise ValueError('Empty split: ' + split)
        files = [str(row[0]) for row in rows]
        labels = {int(row[1]) for row in rows}
        if not labels <= set(range(manifest['classes'])):
            raise ValueError('Labels out of range in ' + split)
        for filename in files:
            stem = filename.split('.')[0]
            for path in [root / 'jpg_video_ori' / split / stem / 'img_00001.jpg',
                         root / 'Keypoints_2d_mmpose' / split / (stem + '.pkl')]:
                if not path.is_file():
                    raise FileNotFoundError('Missing file or broken relocated link: ' + str(path))
        normalized = '\n'.join('{} {}'.format(row[0], row[1]) for row in rows).encode()
        details[split] = dict(rows=len(rows), classes=len(labels), duplicate_excess=len(files) - len(set(files)),
                              effective_list_sha256=hashlib.sha256(normalized).hexdigest())
        file_sets[split] = set(files)
    for left, right in [('train', 'val'), ('train', 'test'), ('val', 'test')]:
        if file_sets[left] & file_sets[right]:
            raise ValueError('Exact clips cross splits: {}/{}'.format(left, right))
    # Exercise actual sample conversion in every split, without a full hour-long scan.
    for base, indexes in [(train, [0, train.flag - 1, train.flag, len(train) - 1]), (test, [0, len(test) - 1])]:
        for index in indexes:
            sample = base.get_sample(index)
            for tensor in [sample['right']['kp2d'], sample['left']['kp2d'], sample['body']['body_pose']]:
                if not torch.isfinite(tensor).all():
                    raise ValueError('Nonfinite sample at index ' + str(index))
    checkpoint = Path(manifest['pretrained_checkpoint'])
    from training.study_state import trusted_load
    payload = trusted_load(checkpoint)
    state = payload.get('state_dict', payload)
    if not any(key.startswith(('encoder_q.', 'module.encoder_q.')) for key in state):
        raise ValueError('Pretrained checkpoint does not contain MASA encoder_q weights.')
    # Run a disposable dense forward/backward using actual prepared inputs.
    # It runs in this subprocess and cannot change the later training RNG/state.
    from moco.builder_dist import MASA
    model = MASA(skeleton_representation='graph-based', num_class=manifest['classes'],
                 pretrain=False, dropout=None)
    current = model.state_dict()
    state = {key[7:] if key.startswith('module.') else key: value for key, value in state.items()}
    compatible = {key: value for key, value in state.items()
                  if key in current and value.shape == current[key].shape}
    if not any(key.startswith('encoder_q.GCN_Tran.') for key in compatible):
        raise ValueError('Pretraining has no compatible transformer weights.')
    current.update(compatible)
    model.load_state_dict(current)
    del payload, state, current, compatible
    model = model.cuda().train()
    inputs, labels = collate([eval_dataset[0], eval_dataset[min(1, len(eval_dataset) - 1)]])
    logits = model({key: value.cuda() for key, value in inputs.items()})
    if logits.shape != (2, manifest['classes']) or not torch.isfinite(logits).all():
        raise ValueError('Model/data smoke test produced invalid logits.')
    loss = torch.nn.functional.cross_entropy(logits, labels.cuda())
    loss.backward()
    for parameter in model.parameters():
        if parameter.grad is not None and not torch.isfinite(parameter.grad).all():
            raise ValueError('Model/data smoke test produced nonfinite gradients.')
    torch.cuda.synchronize()
    del model, loss, logits
    torch.cuda.empty_cache()
    from analysis.benchmark_study import preflight_flop_layouts
    flop_preflight = preflight_flop_layouts(inputs, manifest['classes'], torch.device('cuda'))
    versions = {}
    for package in ('torch', 'torchvision', 'numpy', 'scipy', 'pillow', 'h5py', 'tqdm', 'termcolor'):
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = None
    metadata_hashes = {}
    for folder in (root / 'traintestlist', root / 'Annotations'):
        if folder.is_dir():
            for path in sorted(folder.rglob('*list01.txt')):
                metadata_hashes[str(path.relative_to(root))] = sha256(path)
    for path in sorted(root.glob('*.json')):
        metadata_hashes[path.name] = sha256(path)
    driver = subprocess.check_output(
        ['nvidia-smi', '--id=' + uuid, '--query-gpu=driver_version', '--format=csv,noheader'], text=True
    ).strip()
    return dict(gpu_uuid=uuid, gpu=torch.cuda.get_device_name(0), cuda=torch.version.cuda,
                python=sys.version, driver=driver, cudnn=torch.backends.cudnn.version(),
                training_backend_defaults=dict(matmul_tf32=torch.backends.cuda.matmul.allow_tf32,
                                               cudnn_tf32=torch.backends.cudnn.allow_tf32,
                                               cudnn_benchmark=torch.backends.cudnn.benchmark),
                packages=versions, splits=details, metadata_sha256=metadata_hashes,
                pretrained_sha256=sha256(checkpoint),
                dataset_hash_scope='Effective split lists and metadata, not every pose/frame content.',
                sample_check_scope='First/last train, val, test samples; first image and pose presence for all clips.',
                smoke_test='Dense pretrained model, real batch of 2, CUDA forward and backward.',
                flop_preflight=flop_preflight)


# Result comparisons and export

RESOURCE_KEYS = ('parameter_count', 'latency_ms_per_batch', 'model_size_mb', 'gflops_per_sample')


def save_predictions(loader, rows, state_path):
    """Record the actual sequential evaluation outputs, outside timed forwards."""
    from torch.utils.data import SequentialSampler
    if not isinstance(loader.sampler, SequentialSampler):
        raise ValueError('Prediction recording requires sequential evaluation order.')
    clips = loader.dataset.base_ds.video_list
    if len(rows) != len(clips):
        raise ValueError('Prediction count does not match the effective test list.')
    records = []
    for index, (target, prediction) in enumerate(rows):
        clip = str(clips[index][0])
        records.append(dict(index=index, clip=clip, target=target, prediction=prediction,
                            correct=target == prediction))
    write_json(Path(state_path).with_suffix('.predictions.json'),
               dict(count=len(records), correct=sum(row['correct'] for row in records), records=records))


def deltas(model, reference):
    values = {'accuracy_pp': (model['accuracy'] - reference['accuracy']) * 100.0}
    for key in RESOURCE_KEYS:
        values[key + '_relative_percent'] = (model[key] / reference[key] - 1.0) * 100.0
    return values


def export(manifest_path, benchmark_path, directory):
    manifest = read_json(manifest_path)
    benchmark = read_json(benchmark_path)
    if manifest['dataset'] != benchmark['dataset']:
        raise ValueError('Benchmark dataset mismatch.')
    expected = {entry['id'] for entry in manifest['models']}
    if expected != set(benchmark['models']) or len(expected) != 14:
        raise ValueError('Expected exactly 14 matching benchmark models.')
    rows = []
    prediction_vectors = {}
    sample_identity = None
    for entry in manifest['models']:
        report = read_json(entry['report'])
        if Path(report['checkpoint']).resolve() != Path(entry['checkpoint']).resolve():
            raise ValueError('Evaluation checkpoint mismatch: ' + entry['id'])
        accuracy = (report['after_int8_supported_fp16_fallback'] if entry['quantized'] else report['metrics'])['accuracy']
        measured = benchmark['models'][entry['id']]
        audit = measured.get('flop_audit', {})
        if audit.get('unsupported_operators') != {} or not audit.get('by_operator'):
            raise ValueError('Missing or incomplete FLOP audit: ' + entry['id'])
        flop_total = sum(audit['by_operator'].values())
        if not math.isclose(measured['gflops_per_sample'], flop_total / measured['batch_size'] / 1e9,
                            rel_tol=1e-12, abs_tol=0):
            raise ValueError('GFLOPs does not match audited operator totals: ' + entry['id'])
        path = entry.get('quantized_checkpoint', entry['checkpoint'])
        if measured['checkpoint_sha256'] != sha256(path):
            raise ValueError('Benchmark checkpoint changed: ' + entry['id'])
        metrics = {'accuracy': accuracy, **{key: measured[key] for key in RESOURCE_KEYS}}
        predictions = read_json(entry['predictions'])
        records = predictions['records']
        identity = [(row['index'], row['clip'], row['target']) for row in records]
        if not records or (sample_identity is not None and identity != sample_identity):
            raise ValueError('Evaluation sample identities differ: ' + entry['id'])
        sample_identity = identity
        vector = [row['prediction'] for row in records]
        correct = sum(row['prediction'] == row['target'] for row in records)
        if not math.isclose(correct / len(records), accuracy, rel_tol=0, abs_tol=1e-12):
            raise ValueError('Predictions do not reproduce reported accuracy: ' + entry['id'])
        prediction_vectors[entry['id']] = vector
        if any(not math.isfinite(value) for value in metrics.values()):
            raise ValueError('Nonfinite metric: ' + entry['id'])
        rows.append(dict(id=entry['id'], precision=measured['precision'], metrics=metrics,
                         latency_p25_ms=measured['p25_ms'], latency_p75_ms=measured['p75_ms'],
                         batch_size=measured['batch_size'], source_report=entry['report'],
                         source_report_sha256=sha256(entry['report']), predictions=entry['predictions'],
                         predictions_sha256=sha256(entry['predictions'])))
    by_id = {row['id']: row for row in rows}
    baseline = by_id['baseline_fp32']['metrics']
    for row in rows:
        row['vs_baseline'] = deltas(row['metrics'], baseline)
        if '_ghost_kd_' in row['id']:
            # Recovery comparison uses the original FP32 Ghost student.
            reference_id = row['id'].split('_ghost_kd_')[0] + '_ghost_fp32'
            row['vs_pre_kd_ghost'] = deltas(row['metrics'], by_id[reference_id]['metrics'])
    directory.mkdir(parents=True, exist_ok=True)
    pairs = []
    for index, first in enumerate(rows):
        for second in rows[index + 1:]:
            a, b = first['id'], second['id']
            pairs.append(dict(first=a, second=b, same_accuracy=first['metrics']['accuracy'] == second['metrics']['accuracy'],
                              prediction_disagreements=sum(x != y for x, y in zip(prediction_vectors[a], prediction_vectors[b]))))
    write_json(directory / 'results.json', dict(dataset=manifest['dataset'], baseline='baseline_fp32',
                                               benchmark=str(benchmark_path),
                                               flop_convention=benchmark.get('flop_convention'),
                                               standardized_metrics=RESOURCE_KEYS, results=rows,
                                               prediction_comparisons=pairs))
    fields = ['experiment', 'accuracy_percent'] + list(RESOURCE_KEYS) + ['accuracy_change_pp'] + [key + '_change_percent' for key in RESOURCE_KEYS]
    with (directory / 'results.csv').open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            flat = dict(experiment=row['id'], accuracy_percent=row['metrics']['accuracy'] * 100,
                        accuracy_change_pp=row['vs_baseline']['accuracy_pp'])
            flat.update({key: row['metrics'][key] for key in RESOURCE_KEYS})
            flat.update({key + '_change_percent': row['vs_baseline'][key + '_relative_percent'] for key in RESOURCE_KEYS})
            writer.writerow(flat)
    print('[PASS] Exported 14 standardized result rows to ' + str(directory), flush=True)


# Transferred dataset links

def repair(workspace, apply=False):
    mappings = {
        '/workspace/WLASL/latest_full_wlasl': workspace / 'datasets/WLASL/latest_full_wlasl',
        '/workspace/MSASL/native_masa_ready': workspace / 'datasets/MSASL/native_masa_ready',
    }
    proposed, broken = [], []
    for root in mappings.values():
        if not root.is_dir():
            raise FileNotFoundError(root)
        # Path.rglob does not descend through directory symlinks.
        for link in root.rglob('*'):
            if not link.is_symlink():
                continue
            original = os.readlink(link)
            replacement = None
            for old, new in mappings.items():
                if original == old or original.startswith(old + '/'):
                    replacement = str(new) + original[len(old):]
                    break
            if replacement is not None:
                if not Path(replacement).exists():
                    broken.append((str(link), replacement))
                    continue
                proposed.append((str(link), original, replacement))
            elif not link.exists():
                broken.append((str(link), original))
    print('Known absolute links to relocate:', len(proposed))
    print('Unresolved targets:', len(broken))
    for record in (proposed + broken)[:10]:
        print(record)
    if broken:
        raise RuntimeError('Resolve missing/unmapped targets before applying repairs.')
    if apply:
        for path, original, replacement in proposed:
            link = Path(path)
            if os.readlink(link) != original:
                raise RuntimeError('Link changed during scan: ' + path)
            temporary = link.with_name(link.name + '.relocation_tmp')
            temporary.symlink_to(replacement)
            os.replace(temporary, link)
        print('Repaired', len(proposed), 'links; relative links and regular files unchanged.')
    else:
        print('Preview only. Add --apply to perform the listed repairs.')


# Study execution

def source_record(repo):
    files = []
    for directory in ('training', 'analysis', 'experiments', 'data_prep', 'feeder', 'moco', 'utils'):
        files.extend((repo / directory).rglob('*.py'))
    hashes = {str(path.relative_to(repo)): sha256(path) for path in sorted(files)}
    revision = subprocess.check_output(['git', '-C', str(repo), 'rev-parse', 'HEAD'], text=True).strip()
    return dict(git_revision=revision, source_sha256=hashes)


def signatures(paths):
    return {path: dict(bytes=Path(path).stat().st_size, sha256=sha256(path)) for path in paths}


def execute(command, log, env):
    log.parent.mkdir(parents=True, exist_ok=True)
    with log.open('a') as handle:
        handle.write('\n[COMMAND] ' + shlex.join(command) + '\n')
        handle.flush()
        process = subprocess.Popen(command, stdout=handle, stderr=subprocess.STDOUT,
                                   env=env, start_new_session=True)
        previous = {}

        def terminate(signum, frame):
            if process.poll() is None:
                os.killpg(process.pid, signum)
                try:
                    process.wait(timeout=15)
                except subprocess.TimeoutExpired:
                    os.killpg(process.pid, signal.SIGKILL)
                    process.wait()
            raise SystemExit(128 + signum)

        for sig in (signal.SIGINT, signal.SIGTERM):
            previous[sig] = signal.signal(sig, terminate)
        try:
            status = process.wait()
        finally:
            for sig, handler in previous.items():
                signal.signal(sig, handler)
        if status:
            raise RuntimeError('Command exited {}. See {}'.format(status, log))


def validate_outputs(task):
    for path in task.outputs:
        if not Path(path).is_file() or Path(path).stat().st_size == 0:
            raise RuntimeError('Task did not create its required artifact: ' + path)
    if task.kind == 'train':
        history = read_json(task.outputs[2])
        if [row['epoch'] for row in history] != list(range(1, 61)):
            raise RuntimeError('Stage did not finish all 60 epochs: ' + task.name)
        summary = read_json(task.outputs[1])
        best = max(history, key=lambda row: row['val_acc'])
        if summary['best_epoch'] != best['epoch']:
            raise RuntimeError('Selected epoch differs from validation-best: ' + task.name)
    if task.kind in ('evaluate', 'quantize'):
        report = read_json(task.outputs[0])
        values = report['metrics'] if task.kind == 'evaluate' else report['after_int8_supported_fp16_fallback']
        if not 0 <= values['accuracy'] <= 1:
            raise RuntimeError('Invalid evaluation accuracy: ' + task.name)
        if report.get('load_info', {}).get('skipped', 0) != 0:
            raise RuntimeError('Evaluation skipped checkpoint tensors: ' + task.name)


def remove_measurement_copies(task):
    if task.kind == 'evaluate':
        report = Path(task.outputs[0])
        temporary = report.with_name(report.stem + '_state_dict.pth')
    elif task.kind == 'quantize':
        temporary = Path(task.outputs[0]).parent / 'before_fp32_state_dict.pth'
    else:
        return
    # These exact copies were created only to measure serialization size. The
    # original FP32 checkpoint and actual quantized state remain preserved.
    if temporary.is_file():
        temporary.unlink()


def run(args):
    with ExitStack() as stack:
        return _run(args, stack)


def _run(args, stack):
    repo = Path(__file__).resolve().parents[1]
    dataset = args.family + str(args.classes)
    output = args.results_root.resolve() / dataset
    tasks, manifest = build_plan(repo, sys.executable, args.workspace.resolve(), args.family,
                                 args.classes, output, args.workers, args.benchmark_warmup,
                                 args.benchmark_repetitions, args.benchmark_rounds)
    if args.dry_run:
        print(json.dumps(dict(manifest=manifest, tasks=[asdict(task) for task in tasks]), indent=2))
        return
    if args.workers < 0:
        raise ValueError('Workers cannot be negative.')
    assigned_cpus = os.environ.get('SLURM_CPUS_PER_TASK')
    if assigned_cpus and args.workers > int(assigned_cpus):
        raise ValueError('Workers exceed Slurm CPU allocation. Request more CPUs or explicitly set --workers.')
    if output.exists() and any(output.iterdir()) and not args.resume:
        raise ValueError('Output exists. Use --resume for this study or a new --results-root.')
    output.mkdir(parents=True, exist_ok=True)
    lock = stack.enter_context((output / '.run.lock').open('a'))
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    # Acquire GPU ownership before preflight can run its scratch model.
    uuid = gpu_uuid()
    gpu_lock_path = Path(args.workspace) / 'logs' / (uuid + '.study.lock')
    gpu_lock_path.parent.mkdir(parents=True, exist_ok=True)
    gpu_lock = stack.enter_context(gpu_lock_path.open('a'))
    fcntl.flock(gpu_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    # Serialize the shared protocol write across concurrent dataset launches.
    protocol_lock = stack.enter_context((output.parent / '.protocol.lock').open('a'))
    fcntl.flock(protocol_lock, fcntl.LOCK_EX)
    protocol_path = output.parent / 'protocol.json'
    protocol = dict(training_arguments=TRAIN, low_rank_ratio=0.125, low_rank_targets=LOW_TARGETS,
                    low_rank_min_features=64, ghost_mode='all', ghost_ratio=2, kd_alpha=0.5, kd_temperature=4.0,
                    stages='Method 1 dense 60 then compressed 60; Method 2 compressed 60+60; Ghost KD adds 60.',
                    model_variants_per_dataset=14,
                    workers='Explicitly recorded per dataset; must fit allocation.',
                    benchmark=dict(batch_size=32, target_t=32, warmup=args.benchmark_warmup,
                                   repetitions=args.benchmark_repetitions, rounds=args.benchmark_rounds,
                                   tf32=False, timing='Synchronized GPU-resident forward; median and variability.',
                                   flops='Audited JIT operation trace; MAC=2; GFLOPs/sample normalized from batch 32; unknown operators fatal.'))
    if protocol_path.exists() and read_json(protocol_path) != protocol:
        raise ValueError('Shared study protocol differs. Use a new results root.')
    write_json(protocol_path, protocol)
    fcntl.flock(protocol_lock, fcntl.LOCK_UN)
    env = dict(os.environ, PYTHONPATH=str(repo) + os.pathsep + os.environ.get('PYTHONPATH', ''),
               PYTHONUNBUFFERED='1', MASA_STUDY_PREDICTIONS='1')
    env['MASA_STUDY_LOCK_OWNER'] = str(os.getpid())
    env['MASA_STUDY_LOCK_GPU'] = uuid
    plan = dict(tasks=[asdict(task) for task in tasks], manifest=manifest,
                configuration=dict(family=args.family, classes=args.classes, workers=args.workers,
                                   benchmark_warmup=args.benchmark_warmup,
                                   benchmark_repetitions=args.benchmark_repetitions,
                                   benchmark_rounds=args.benchmark_rounds))
    plan_path = output / 'plan.json'
    if plan_path.exists() and read_json(plan_path) != plan:
        raise ValueError('Saved study plan differs. Use a new results directory.')
    source = source_record(repo)
    source_path = output / 'source.json'
    if source_path.exists() and read_json(source_path)['source_sha256'] != source['source_sha256']:
        raise ValueError('Study source changed since launch. Restore it or use a new results directory.')
    write_json(plan_path, plan)
    if not source_path.exists():
        write_json(source_path, source)
    write_json(output / 'models.json', manifest)
    print('[PREFLIGHT] {}: checking files, splits, sample loading, and GPU'.format(dataset), flush=True)
    execute([sys.executable, str(repo / 'experiments/run_study.py'),
             '--preflight-manifest', str(output / 'models.json'), '--out', str(output / 'preflight.current.json')],
            output / 'logs/preflight.log', env)
    current = read_json(output / 'preflight.current.json')
    preflight_path = output / 'preflight.json'
    if preflight_path.exists():
        old = read_json(preflight_path)
        for key in ('splits', 'metadata_sha256', 'pretrained_sha256', 'packages', 'cuda', 'gpu',
                    'python', 'driver', 'cudnn', 'training_backend_defaults'):
            if current[key] != old[key]:
                raise ValueError('Preflight provenance changed: ' + key)
    else:
        write_json(preflight_path, current)
    if current['gpu_uuid'] != uuid:
        raise ValueError('GPU identity changed during preflight.')
    for index, task in enumerate(tasks, 1):
        marker = output / 'task_records' / (task.name + '.json')
        inputs = signatures(task.sources)
        if marker.exists():
            saved = read_json(marker)
            if saved['command'] != task.command or saved['inputs'] != inputs or saved['outputs'] != signatures(task.outputs):
                raise ValueError('Completed task provenance/artifacts changed: ' + task.name)
            print('[SKIP verified] ' + task.name, flush=True)
            continue
        command = list(task.command)
        if task.kind == 'train':
            directory = Path(task.outputs[0]).parent
            config_path = directory / 'launch.json'
            record = dict(command=task.command, inputs=inputs)
            if config_path.exists() and read_json(config_path) != record:
                raise ValueError('Interrupted stage initialization/configuration changed: ' + task.name)
            write_json(config_path, record)
            if (directory / 'last.pth.tar').is_file():
                if not args.resume:
                    raise ValueError('Interrupted stage requires --resume.')
                command.append('--resume-stage')
        print('[START {}/{}] {} | log: {}'.format(index, len(tasks), task.name, output / 'logs' / (task.name + '.log')), flush=True)
        started = time.time()
        execute(command, output / 'logs' / (task.name + '.log'), env)
        validate_outputs(task)
        remove_measurement_copies(task)
        write_json(marker, dict(command=task.command, executed_command=command, inputs=inputs,
                                outputs=signatures(task.outputs), elapsed_seconds=time.time() - started))
        print('[DONE] ' + task.name, flush=True)
    print('[COMPLETE] ' + str(output / 'comparisons/results.csv'), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('classes', type=int, nargs='?')
    parser.add_argument('--family', choices=('wlasl', 'msasl'))
    parser.add_argument('--workspace', type=Path, default=Path('/shared/g00107537/jupyter'))
    parser.add_argument('--results-root', type=Path)
    parser.add_argument('--workers', type=int, default=8)
    parser.add_argument('--benchmark-warmup', type=int, default=30)
    parser.add_argument('--benchmark-repetitions', type=int, default=100)
    parser.add_argument('--benchmark-rounds', type=int, default=3)
    parser.add_argument('--dry-run', action='store_true')
    parser.add_argument('--resume', action='store_true')
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument('--preflight-manifest', type=Path, help=argparse.SUPPRESS)
    modes.add_argument('--export-manifest', type=Path, help=argparse.SUPPRESS)
    modes.add_argument('--repair-links', action='store_true', help='Preview transferred RunPod symlink repairs.')
    parser.add_argument('--apply', action='store_true', help='Apply --repair-links after validating targets.')
    parser.add_argument('--out', type=Path, help=argparse.SUPPRESS)
    parser.add_argument('--benchmark', type=Path, help=argparse.SUPPRESS)
    parser.add_argument('--out-dir', type=Path, help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.preflight_manifest:
        if args.out is None:
            parser.error('Preflight requires --out.')
        write_json(args.out, preflight(read_json(args.preflight_manifest)))
        print('[PASS] Dataset/GPU preflight complete.', flush=True)
        return
    if args.export_manifest:
        if args.benchmark is None or args.out_dir is None:
            parser.error('Export requires --benchmark and --out-dir.')
        export(args.export_manifest, args.benchmark, args.out_dir)
        return
    if args.repair_links:
        repair(args.workspace, args.apply)
        return
    if args.apply:
        parser.error('--apply requires --repair-links.')
    if args.classes is None or args.family is None:
        parser.error('A dataset family and class-count variant are required.')
    if args.results_root is None:
        args.results_root = args.workspace / 'results/compression_study_v1'
    run(args)


if __name__ == "__main__":
    main()
