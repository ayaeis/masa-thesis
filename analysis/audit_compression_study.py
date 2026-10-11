#!/usr/bin/env python3
"""Read-only audit of the final study's existing result artifacts.

No models are trained or evaluated. Missing historical evidence is never PASS.
Paths may be relocated, but the dataset-relative checkpoint path must match.
"""

import argparse
import json
import math
from pathlib import Path


DATASETS = ('wlasl100', 'wlasl300', 'msasl100', 'msasl200', 'msasl1000')
METRICS = ('accuracy', 'param_count_tensors', 'latency_ms_per_batch',
           'model_size_mb', 'flops_per_batch')


def read(path):
    return json.loads(path.read_text())


def same_path(actual, expected, dataset):
    # Preserve the dataset and every subsequent component, even after relocation.
    parts = Path(str(actual)).parts
    wanted = expected.parts
    if dataset not in parts or dataset not in wanted:
        return False
    return parts[parts.index(dataset):] == wanted[wanted.index(dataset):]


def audit(dataset, label, ckpt, report_path, quant, baseline, parent, kind):
    errors, gaps = [], []
    required = [ckpt, report_path]
    if not quant:
        required += [ckpt.parent / 'summary.json', ckpt.parent / 'metrics.json']
    missing = [str(p) for p in required if not p.is_file()]
    if missing:
        return dict(dataset=dataset, experiment=label, status='MISSING_ARTIFACTS',
                    missing=missing)
    try:
        report = read(report_path)
        if not same_path(report.get('checkpoint', ''), ckpt, dataset):
            errors.append('Evaluation/quantization checkpoint does not match experiment.')
        values = report.get('after_int8_supported_fp16_fallback' if quant else 'metrics', {})
        for key in METRICS:
            val = values.get(key)
            if not isinstance(val, (int, float)) or not math.isfinite(val):
                errors.append('Missing/nonfinite metric: ' + key)
        if isinstance(values.get('accuracy'), (int, float)) and not 0 <= values['accuracy'] <= 1:
            errors.append('Accuracy outside [0, 1].')
        if not quant:
            if report.get('num_class') != int(''.join(c for c in dataset if c.isdigit())):
                errors.append('Wrong evaluation class count.')
            if kind in ('ghost', 'kd'):
                for key, expected in [('use_ghost_conv', True), ('ghost_ratio', 2), ('ghost_mode', 'all')]:
                    if report.get(key) != expected:
                        errors.append('Wrong Ghost setting: ' + key)
            if kind == 'lowrank' or (quant and kind == 'lowrank'):
                for key, expected in [('use_low_rank', True), ('low_rank_ratio', 0.125), ('low_rank_min_features', 64)]:
                    if report.get(key) != expected:
                        errors.append('Wrong low-rank setting: ' + key)
                if set(str(report.get('low_rank_targets', '')).split(',')) != {'transformer', 'conv', 'project_head'}:
                    errors.append('Wrong low-rank target layers.')
            summary = read(ckpt.parent / 'summary.json')
            history = read(ckpt.parent / 'metrics.json')
            if not isinstance(history, list) or [r.get('epoch') for r in history] != list(range(1, 61)):
                errors.append('Training history is not a complete 60-epoch stage.')
            else:
                expected_best = max(history, key=lambda r: r['val_acc'])
                if summary.get('best_epoch') != expected_best['epoch']:
                    errors.append('Best epoch does not match first maximum validation accuracy.')
                for row in history:
                    expected_lr = 0.01 * (0.1 ** (int(row['epoch'] >= 20) + int(row['epoch'] >= 40)))
                    if not math.isclose(row.get('lr', -1), expected_lr, rel_tol=1e-6, abs_tol=1e-10):
                        errors.append('Saved learning-rate history differs from 0.01, milestones 20/40, gamma 0.1.')
                        break
            if not same_path(summary.get('best_checkpoint', ''), ckpt, dataset):
                errors.append('Training summary checkpoint does not match experiment.')
            if not math.isclose(summary.get('test_acc', -1), values.get('accuracy', -2), abs_tol=1e-8):
                errors.append('Training-summary and evaluation accuracies differ; investigate evaluation settings.')
            if parent:
                source_key = 'student_init_ckpt' if kind == 'kd' else ('dense_ckpt' if 'overnight_final_run' in ckpt.parts and kind == 'lowrank' else 'init_ckpt')
                if source_key not in summary:
                    gaps.append('Initialization checkpoint was not saved in summary (' + source_key + ').')
                elif parent == 'pretraining':
                    gaps.append('Recorded pretraining path needs original checkpoint/hash verification: ' + str(summary[source_key]))
                elif not same_path(summary[source_key], parent, dataset):
                    errors.append('Wrong initialization checkpoint: ' + source_key)
            if kind == 'kd':
                if not same_path(summary.get('teacher_ckpt', ''), baseline, dataset):
                    errors.append('KD teacher is not this dataset baseline.')
                for key, expected in [('kd_alpha', 0.5), ('kd_temp', 4.0), ('teacher_use_ghost_conv', False), ('student_ghost_mode', 'all')]:
                    if summary.get(key) != expected:
                        errors.append('Wrong KD setting: ' + key)
            gaps.append('Verify historical batch size 64, SGD/momentum 0.9, weight decay 0, seed 123, target_t 32, index sampling, and remaining arguments from original launch records.')
        else:
            gaps.append('Verify quantization launch architecture, data root, and precision settings from original records.')
        if 'baseline' in report and not same_path(report['baseline'].get('checkpoint', ''), baseline, dataset):
            errors.append('Wrong evaluation baseline checkpoint.')
        gaps.append('Historical dataset/split/preprocessing hashes and source checkpoint hashes were not established by this audit.')
        return dict(dataset=dataset, experiment=label,
                    status='MISMATCH' if errors else 'NEEDS_EVIDENCE',
                    errors=errors, evidence_gaps=gaps, report=str(report_path))
    except (ValueError, TypeError, KeyError, OSError) as exc:
        return dict(dataset=dataset, experiment=label, status='MISMATCH', errors=[str(exc)])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--results-root', type=Path, default=Path('fall_results'))
    parser.add_argument('--out', type=Path, default=Path('compression_study_audit.json'))
    args = parser.parse_args()
    rows = []
    for ds in DATASETS:
        root = args.results_root / ds
        base = root / 'baseline' / 'best.pth.tar'
        post = root / 'overnight_final_run'
        ghost = post / 'ghost_allk' / 'best.pth.tar'
        kd = post / 'kd_ghost_allk' / 'best.pth.tar'
        low = post / 'lowrank' / 'all_r0125' / 'best.pth.tar'
        specs = [
            ('M1 baseline', base, post / 'reports/baseline_report.json', False, 'pretraining', 'baseline'),
            ('M1 baseline quantized', base, post / 'quant_baseline/summary.json', True, None, 'baseline'),
            ('M1 Ghost', ghost, post / 'reports/ghost_allk_vs_baseline.json', False, base, 'ghost'),
            ('M1 Ghost quantized', ghost, post / 'quant_ghost_allk/summary.json', True, None, 'ghost'),
            ('M1 Ghost KD', kd, post / 'reports/kd_ghost_allk_vs_baseline.json', False, ghost, 'kd'),
            ('M1 Ghost KD quantized', kd, post / 'quant_kd_ghost_allk/summary.json', True, None, 'kd'),
            ('M1 low-rank', low, low.parent / 'report_vs_baseline.json', False, base, 'lowrank'),
            ('M1 low-rank quantized', low, low.parent / 'quantized/summary.json', True, None, 'lowrank'),
        ]
        arch = root / 'architecture_first'
        for folder, kind in [('ghost_allk', 'ghost'), ('lowrank_all_r0125', 'lowrank')]:
            for stage in (60, 120):
                ck = arch / folder / ('epoch' + str(stage)) / 'best.pth.tar'
                parent = 'pretraining' if stage == 60 else arch / folder / 'epoch60/best.pth.tar'
                specs.append(('M2 ' + folder + ' stage ' + str(stage), ck,
                              arch / 'reports' / (folder + '_epoch' + str(stage) + '_vs_baseline.json'), False, parent, kind))
        for label, ck, report, quant, parent, kind in specs:
            row = audit(ds, label, ck, report, quant, base, parent, kind)
            rows.append(row)
            print('{:<10} {:<35} {}'.format(ds, label, row['status']))
            for error in row.get('errors', []):
                print('  ' + error)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps({'scope': 'Existing M1 artifacts and M2 training stages; no historical protocol certification.', 'experiments': rows}, indent=2) + '\n')
    print('Detailed evidence gaps: ' + str(args.out))
    print('NEEDS_EVIDENCE is not a rerun decision or a full protocol PASS.')


if __name__ == '__main__':
    main()
