"""CPU checks for the sequential low-rank KD extension and exact factor loading."""

import ast
import copy
import csv
import json
import io
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
from contextlib import ExitStack, redirect_stdout

import torch

from experiments import run_lowrank_kd_study as extension
from experiments.run_study import build_plan, export, read_json, sha256, write_json
from moco.low_rank_modules import apply_low_rank_structure

ROOT = Path(__file__).resolve().parents[1]


class TinyModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder_q = torch.nn.Module()
        self.encoder_q.GCN_Tran = torch.nn.Sequential(torch.nn.Linear(8, 8))

    def forward(self, inputs):
        return self.encoder_q.GCN_Tran(inputs['features'])


class ExtensionPlanTests(unittest.TestCase):
    def plan(self, dataset='wlasl100'):
        return extension.build_extension_plan(ROOT, '/python', '/workspace', '/old', '/new', dataset)

    def test_all_six_datasets_have_exactly_two_new_training_runs_and_four_variants(self):
        training_count = 0
        for dataset in extension.DATASETS:
            tasks, manifest = self.plan(dataset)
            self.assertEqual(len(tasks), 8)
            self.assertEqual(len(manifest['models']), 18)
            self.assertEqual(len({m['id'] for m in manifest['models']}), 18)
            self.assertEqual(len(manifest['training_checkpoints']), 11)
            training = [t for t in tasks if t.kind == 'train']
            training_count += len(training)
            self.assertEqual(len(training), 2)
            by_id = {m['id']: m for m in manifest['models']}
            family = manifest['family']
            _, old = build_plan(ROOT, '/python', '/workspace', family, manifest['classes'],
                                Path('/old') / dataset, 4)
            self.assertEqual(manifest['models'][:14], old['models'])
            for method, task in zip(('m1', 'm2'), training):
                command = task.command
                initial = command[command.index('--student-ckpt') + 1]
                self.assertEqual(initial, by_id[method + '_lowrank_fp32']['checkpoint'])
                self.assertTrue(initial.endswith('/fp32/best.pth.tar') if method == 'm1'
                                else initial.endswith('/stage2/best.pth.tar'))
                self.assertEqual(command[command.index('--teacher-ckpt') + 1],
                                 by_id['baseline_fp32']['checkpoint'])
                self.assertIn('--student-use-low-rank', command)
                self.assertNotIn('--student-use-ghost-conv', command)
                self.assertIn('--save-last', command)
                self.assertTrue(all(p.startswith('/new/') for p in task.outputs))
                self.assertEqual(command[command.index('--dropout') + 1], '0.0')
                # Common training arguments are byte-for-byte the Ghost KD protocol.
                start = command.index('--epochs')
                self.assertEqual(command[start:start + len(extension.TRAIN)], extension.TRAIN)
            self.assertEqual(tasks[-2].command[tasks[-2].command.index('--warmup') + 1], '30')
            self.assertEqual(tasks[-2].command[tasks[-2].command.index('--repetitions') + 1], '100')
            self.assertEqual(tasks[-2].command[tasks[-2].command.index('--rounds') + 1], '3')
        self.assertEqual(training_count, 12)

    def test_initialization_must_be_stage2_and_original_manifest_is_not_mutated(self):
        _, original = build_plan(ROOT, '/python', '/workspace', 'wlasl', 100, Path('/old/wlasl100'), 4)
        before = copy.deepcopy(original)
        extension.build_extension_plan(ROOT, '/python', '/workspace', '/old', '/new', 'wlasl100', original=original)
        self.assertEqual(original, before)
        original['models'][12]['checkpoint'] = original['training_checkpoints']['m2_lowrank_stage1_train']
        with self.assertRaisesRegex(ValueError, 'intended training stage'):
            extension.build_extension_plan(ROOT, '/python', '/workspace', '/old', '/new', 'wlasl100', original=original)

    def test_output_root_cannot_overwrite_or_nest_originals(self):
        for source, output in [('/old', '/old'), ('/old', '/old/kd'), ('/old/study', '/old')]:
            with self.assertRaises(ValueError):
                extension.check_roots(source, output)
        extension.check_roots('/results/old', '/results/new')

    def test_all_generated_training_arguments_parse_with_correct_family(self):
        for dataset in extension.DATASETS:
            tasks, manifest = self.plan(dataset)
            module = extension.import_kd_trainer(manifest['family'])
            for task in tasks:
                if task.kind == 'train':
                    with patch.object(sys, 'argv', task.command[1:]):
                        args = module.parse_args()
                    self.assertEqual(args.num_class, manifest['classes'])
                    self.assertTrue(args.student_use_low_rank)
                    self.assertEqual(args.workers, 4)
                    self.assertEqual(args.kd_alpha, 0.5)
                    self.assertEqual(args.kd_temp, 4.0)
                    self.assertEqual(args.student_low_rank_ratio, 0.125)


class ExactStudentTests(unittest.TestCase):
    def fixture(self, root):
        model = TinyModel()
        apply_low_rank_structure(model, 0.125, 'transformer,conv,project_head', 1)
        metadata = dict(rank_ratio=0.125, low_rank_targets='transformer,conv,project_head',
                        low_rank_min_features=1)
        path = root / 'best.pth.tar'
        torch.save(dict(state_dict={'module.' + k: v for k, v in model.state_dict().items()}, **metadata), path)
        args = SimpleNamespace(num_class=8, dropout=0.0, student_use_ghost_conv=False,
                               teacher_use_ghost_conv=False, student_low_rank_ratio=0.125,
                               student_low_rank_targets=metadata['low_rank_targets'],
                               student_low_rank_min_features=1, student_ckpt=str(path))
        return model, metadata, path, args

    def test_factors_restore_exactly_without_svd_for_both_families(self):
        for family in ('wlasl', 'msasl'):
            module = extension.import_kd_trainer(family)
            with tempfile.TemporaryDirectory() as tmp:
                model, metadata, path, args = self.fixture(Path(tmp))
                with patch.object(module, 'build_model', side_effect=lambda *a: TinyModel()), \
                        patch('torch.linalg.svd', side_effect=AssertionError('No SVD during KD')):
                    restored, info, recorded = module.build_low_rank_student(args)
                self.assertEqual(info, dict(loaded=len(model.state_dict()), skipped=0))
                self.assertEqual(recorded, metadata)
                self.assertEqual(sum(p.numel() for p in restored.parameters()),
                                 sum(p.numel() for p in model.parameters()))
                for key, value in model.state_dict().items():
                    self.assertTrue(torch.equal(value, restored.state_dict()[key]), key)

    def test_wrong_rank_dense_weights_missing_factors_and_dtype_are_rejected(self):
        for family in ('wlasl', 'msasl'):
            module = extension.import_kd_trainer(family)
            with tempfile.TemporaryDirectory() as tmp:
                model, metadata, path, args = self.fixture(Path(tmp))
                for failure in ('rank', 'dense', 'missing', 'dtype'):
                    state = dict(model.state_dict())
                    config = dict(metadata)
                    if failure == 'rank':
                        config['rank_ratio'] = 0.25
                    elif failure == 'dense':
                        state = TinyModel().state_dict()
                    elif failure == 'missing':
                        state.pop(next(iter(state)))
                    else:
                        state = {k: v.double() for k, v in state.items()}
                    torch.save(dict(state_dict=state, **config), path)
                    with patch.object(module, 'build_model', side_effect=lambda *a: TinyModel()):
                        with self.assertRaises((ValueError, RuntimeError)):
                            module.build_low_rank_student(args)

    def test_kd_updates_only_student_and_matches_ghost_loss(self):
        for family in ('wlasl', 'msasl'):
            module = extension.import_kd_trainer(family)
            student, teacher = TinyModel(), TinyModel()
            teacher_before = copy.deepcopy(teacher.state_dict())
            student_before = copy.deepcopy(student.state_dict())
            optimizer = torch.optim.SGD(student.parameters(), lr=0.01, momentum=0.9)
            inputs, labels = {'features': torch.randn(4, 8)}, torch.tensor([0, 1, 2, 3])
            with torch.no_grad():
                s, t = student(inputs), teacher(inputs)
                expected = 0.5 * torch.nn.functional.cross_entropy(s, labels) + 0.5 * module.kd_loss_fn(s, t, 4.0)
            result = module.run_train_epoch(student, teacher, [(inputs, labels)],
                                            torch.nn.CrossEntropyLoss(), optimizer, torch.device('cpu'), 0.5, 4.0)
            self.assertAlmostEqual(result.total_loss, expected.item(), places=6)
            self.assertTrue(any(not torch.equal(v, student.state_dict()[k]) for k, v in student_before.items()))
            self.assertTrue(all(torch.equal(v, teacher.state_dict()[k]) for k, v in teacher_before.items()))
            self.assertTrue(all(p.grad is None for p in teacher.parameters()))

    def test_existing_ghost_kd_loss_and_epoch_functions_are_unchanged(self):
        for family in ('wlasl', 'msasl'):
            path = 'training/' + family + '/finetune_ghost_kd.py'
            previous = subprocess.check_output(['git', '-C', str(ROOT), 'show', 'HEAD:' + path], text=True)
            before = {n.name: ast.dump(n) for n in ast.parse(previous).body if isinstance(n, ast.FunctionDef)}
            after = {n.name: ast.dump(n) for n in ast.parse((ROOT / path).read_text()).body if isinstance(n, ast.FunctionDef)}
            for name in ('kd_loss_fn', 'run_train_epoch', 'run_eval_epoch', 'build_model'):
                self.assertEqual(before[name], after[name])

    def test_actual_kd_training_saves_factors_and_resumes_after_interruption(self):
        from training.study_state import StageState, trusted_load

        class Base:
            flag = 2

            def __init__(self, *args, **kwargs):
                self.size = 2 if kwargs['data_split'] == 'test' else 4

            def __len__(self):
                return self.size

        class Samples(torch.utils.data.Dataset):
            def __init__(self, base, **kwargs):
                self.base = base

            def __len__(self):
                return len(self.base)

            def __getitem__(self, index):
                return {'features': torch.arange(8).float() / 8 + index}, index % 8

        def collate(rows):
            return {'features': torch.stack([r[0]['features'] for r in rows])}, torch.tensor([r[1] for r in rows])

        for family in ('wlasl', 'msasl'):
            module = extension.import_kd_trainer(family)
            with tempfile.TemporaryDirectory() as tmp, ExitStack() as stack:
                root = Path(tmp)
                model, metadata, checkpoint, _ = self.fixture(root)
                teacher = root / 'teacher.pth'
                torch.save(dict(state_dict=TinyModel().state_dict()), teacher)
                command = ['trainer', '--data-root', str(root), '--teacher-ckpt', str(teacher),
                           '--student-ckpt', str(checkpoint), '--num-class', '8', '--epochs', '2',
                           '--workers', '0', '--batch-size', '2', '--student-use-low-rank',
                           '--student-low-rank-min-features', '1', '--dropout', '0.0',
                           '--save-last', '--out-dir', str(root / 'kd')]
                stack.enter_context(patch.object(module, 'build_model', side_effect=lambda *a: TinyModel()))
                stack.enter_context(patch.object(module, 'WLASL' if family == 'wlasl' else 'MSASL', Base))
                stack.enter_context(patch.object(module, 'WLASLSupervised' if family == 'wlasl' else 'MSASLSupervised', Samples))
                stack.enter_context(patch.object(module, 'collate_supervised', collate))
                stack.enter_context(patch('torch.cuda.is_available', return_value=False))
                original_save = StageState.save

                def interrupt_after_save(stage, *args):
                    original_save(stage, *args)
                    raise InterruptedError('simulate allocation expiry after epoch one')

                with redirect_stdout(io.StringIO()), patch.object(sys, 'argv', command), \
                        patch.object(StageState, 'save', interrupt_after_save):
                    with self.assertRaises(InterruptedError):
                        module.main()
                self.assertEqual(trusted_load(root / 'kd/last.pth.tar')['epoch'], 1)
                with redirect_stdout(io.StringIO()), patch.object(sys, 'argv', command + ['--resume-stage']):
                    module.main()
                self.assertEqual([r['epoch'] for r in read_json(root / 'kd/metrics.json')], [1, 2])
                best = trusted_load(root / 'kd/best.pth.tar')
                for key, value in metadata.items():
                    self.assertEqual(best[key], value)
                self.assertEqual(set(best['state_dict']), set(model.state_dict()))
                self.assertEqual(best['teacher_ckpt'], str(teacher))
                self.assertEqual(best['student_init_ckpt'], str(checkpoint))


class ExtendedExportTests(unittest.TestCase):
    def test_export_18_variants_and_pre_kd_comparisons(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _, manifest = extension.build_extension_plan(ROOT, '/python', root, root / 'old',
                                                          root / 'new', 'msasl100')
            measured = {}
            for index, entry in enumerate(manifest['models']):
                for key in ('checkpoint', 'quantized_checkpoint'):
                    if key in entry:
                        path = Path(entry[key])
                        path.parent.mkdir(parents=True, exist_ok=True)
                        path.write_bytes(entry['id'].encode())
            for entry in manifest['models']:
                accuracy = 0.5
                report = dict(checkpoint=entry['checkpoint'], metrics=dict(accuracy=accuracy),
                              after_int8_supported_fp16_fallback=dict(accuracy=accuracy))
                write_json(entry['report'], report)
                write_json(entry['predictions'], dict(records=[
                    dict(index=0, clip='a', target=0, prediction=0),
                    dict(index=1, clip='b', target=1, prediction=0)]))
                measured[entry['id']] = dict(parameter_count=10, latency_ms_per_batch=2.0,
                    model_size_mb=1.0, gflops_per_sample=32 / 32 / 1e9, batch_size=32,
                    flop_audit=dict(unsupported_operators={}, by_operator={'aten::mul': 32}),
                    checkpoint_sha256=sha256(entry.get('quantized_checkpoint', entry['checkpoint'])),
                    precision='test', p25_ms=1.9, p75_ms=2.1)
            write_json(root / 'models.json', manifest)
            write_json(root / 'benchmark.json', dict(dataset='msasl100', models=measured))
            export(root / 'models.json', root / 'benchmark.json', root / 'export', expected_count=18)
            results = read_json(root / 'export/results.json')['results']
            self.assertEqual(len(results), 18)
            for row in results:
                if '_lowrank_kd_' in row['id']:
                    self.assertEqual(row['vs_pre_kd_lowrank']['accuracy_pp'], 0)
            with self.assertRaisesRegex(ValueError, '14 matching'):
                export(root / 'models.json', root / 'benchmark.json', root / 'wrong')


class SequentialExecutionTests(unittest.TestCase):
    def test_all_datasets_resume_without_retraining_or_writing_originals(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp).resolve()
            old_root, new_root = root / 'old', root / 'new'
            preflight = {key: 'unchanged' for key in (
                'splits', 'metadata_sha256', 'pretrained_sha256', 'packages', 'cuda', 'python',
                'cudnn', 'training_backend_defaults', 'gpu', 'driver')}

            def report(entry):
                path = Path(entry['checkpoint'])
                if not path.exists():
                    path.parent.mkdir(parents=True, exist_ok=True)
                    path.write_bytes(entry['id'].encode())
                if entry['quantized']:
                    path = Path(entry['quantized_checkpoint'])
                    path.parent.mkdir(parents=True, exist_ok=True)
                    path.write_bytes(entry['id'].encode())
                write_json(entry['report'], dict(checkpoint=entry['checkpoint'], load_info=dict(skipped=0),
                    metrics=dict(accuracy=0.5), after_int8_supported_fp16_fallback=dict(accuracy=0.5)))
                write_json(entry['predictions'], dict(records=[
                    dict(index=0, clip='a', target=0, prediction=0),
                    dict(index=1, clip='b', target=1, prediction=0)]))

            for dataset in extension.DATASETS:
                family = 'wlasl' if dataset.startswith('wlasl') else 'msasl'
                _, manifest = build_plan(ROOT, sys.executable, root, family,
                                         int(dataset[len(family):]), old_root / dataset, 4)
                for stage, filename in manifest['training_checkpoints'].items():
                    path = Path(filename)
                    path.parent.mkdir(parents=True, exist_ok=True)
                    path.write_bytes(stage.encode())
                for entry in manifest['models']:
                    report(entry)
                write_json(old_root / dataset / 'models.json', manifest)
                write_json(old_root / dataset / 'preflight.json', preflight)
                write_json(old_root / dataset / 'comparisons/results.json',
                           dict(dataset=dataset, results=[dict(id=m['id']) for m in manifest['models']]))
                (old_root / dataset / 'comparisons/results.csv').write_text('original export\n')
            before = {str(p): sha256(p) for p in old_root.rglob('*') if p.is_file()}
            executed_train = []

            def execute(command, log, env):
                def argument(flag):
                    return command[command.index(flag) + 1]

                dataset = log.parent.parent.name
                manifest_path = new_root / dataset / 'models.json'
                if '--preflight-manifest' in command:
                    write_json(Path(argument('--out')), preflight)
                elif '--check-initialization' in command:
                    write_json(Path(argument('--out')), dict(smoke='passed'))
                elif '--export-manifest' in command:
                    export(Path(argument('--export-manifest')), Path(argument('--benchmark')),
                           Path(argument('--out-dir')), expected_count=18)
                elif '--teacher-ckpt' in command:
                    executed_train.append(command)
                    directory = Path(argument('--out-dir'))
                    directory.mkdir(parents=True, exist_ok=True)
                    for filename in ('best.pth.tar', 'last.pth.tar'):
                        (directory / filename).write_bytes(str(directory).encode())
                    write_json(directory / 'metrics.json', [dict(epoch=i, val_acc=i / 100) for i in range(1, 61)])
                    write_json(directory / 'summary.json', dict(best_epoch=60))
                elif '--finetuned-ckpt' in command or '--ckpt' in command:
                    manifest = read_json(manifest_path)
                    output = Path(argument('--out-dir')) / 'summary.json' if '--out-dir' in command else Path(argument('--out'))
                    entry = next(m for m in manifest['models'] if m['report'] == str(output))
                    report(entry)
                else:
                    manifest = read_json(Path(argument('--manifest')))
                    measured = {}
                    for entry in manifest['models']:
                        measured[entry['id']] = dict(parameter_count=10, latency_ms_per_batch=2.0,
                            model_size_mb=1.0, gflops_per_sample=1e-9, batch_size=32,
                            flop_audit=dict(unsupported_operators={}, by_operator={'aten::mul': 32}),
                            checkpoint_sha256=sha256(entry.get('quantized_checkpoint', entry['checkpoint'])),
                            precision='test', p25_ms=1.9, p75_ms=2.1)
                    output = Path(argument('--out'))
                    write_json(output, dict(dataset=dataset, models=measured))
                    (output.parent / 'benchmark_inputs.pth').write_bytes(b'fixed batch')
                    write_json(output.parent / 'weight_audit.json', dict(duplicate_groups=[]))

            args = SimpleNamespace(study_root=old_root, output_root=new_root, workspace=root,
                                   datasets=list(extension.DATASETS), workers=4, dry_run=False, resume=False)
            with patch.object(extension, 'execute', side_effect=execute), \
                    patch.object(extension, 'gpu_uuid', return_value='GPU-test'), \
                    patch.object(extension, 'assert_idle'), \
                    patch.object(extension, 'source_record', return_value=dict(source_sha256={'test': 'source'}, git_revision='test')), \
                    redirect_stdout(io.StringIO()):
                extension.run(args)
                self.assertEqual(len(executed_train), 12)
                args.resume = True
                extension.run(args)
                self.assertEqual(len(executed_train), 12)
                # Completed artifact tampering must not be silently skipped.
                (new_root / 'wlasl100/m1/lowrank_all_r0125/kd_fp32/best.pth.tar').write_bytes(b'tampered')
                with self.assertRaisesRegex(ValueError, 'Completed task artifacts'):
                    extension.run(args)
            after = {str(p): sha256(p) for p in old_root.rglob('*') if p.is_file()}
            self.assertEqual(before, after)
            with (new_root / 'all_results.csv').open(newline='') as handle:
                rows = list(csv.DictReader(handle))
            self.assertEqual(len(rows), 108)
            self.assertEqual(len({(r['dataset'], r['experiment']) for r in rows}), 108)


if __name__ == '__main__':
    unittest.main()
