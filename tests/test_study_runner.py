import json
import os
import random
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

from experiments.run_study import deltas, export, sha256, write_json, read_json, build_plan
from training.study_state import StageState, trusted_load


ROOT = Path(__file__).resolve().parents[1]


class PlanTests(unittest.TestCase):
    def plan(self, family='wlasl', classes=100):
        return build_plan(ROOT, '/python', '/workspace', family, classes, Path('/out') / (family + str(classes)))

    def test_all_variants_scope_order_and_initializations(self):
        for family, classes in [('wlasl', 100), ('wlasl', 300), ('msasl', 100), ('msasl', 200), ('msasl', 1000)]:
            tasks, manifest = self.plan(family, classes)
            self.assertEqual(len(tasks), 25)
            self.assertEqual(len(manifest['models']), 14)
            self.assertEqual(sum(task.kind == 'train' for task in tasks), 9)
            self.assertEqual(len(manifest['training_checkpoints']), 9)
            self.assertEqual(len(set(manifest['training_checkpoints'].values())), 9)
            self.assertEqual(sum(task.kind == 'quantize' for task in tasks), 7)
            names = [task.name for task in tasks]
            self.assertLess(names.index('m1_lowrank_quantized'), names.index('m2_ghost_stage1_train'))
            self.assertLess(names.index('m2_ghost_quantized'), names.index('m2_ghost_kd_train'))
            self.assertEqual(names[-2:], ['standardized_benchmark', 'export_comparisons'])
            by_name = {task.name: task for task in tasks}
            baseline = by_name['baseline_train'].outputs[0]
            for prefix in ('m1', 'm2'):
                command = by_name[prefix + '_ghost_kd_train'].command
                self.assertEqual(command[command.index('--teacher-ckpt') + 1], baseline)
                student = command[command.index('--student-ckpt') + 1]
                expected = by_name['m1_ghost_train' if prefix == 'm1' else 'm2_ghost_stage2_train'].outputs[0]
                self.assertEqual(student, expected)
                self.assertNotIn('quantized', student)
            self.assertEqual(by_name['m2_lowrank_stage2_train'].sources,
                             [by_name['m2_lowrank_stage1_train'].outputs[0]])
            for task in tasks:
                self.assertNotIn('finetune_lowrank_kd.py', ' '.join(task.command))
                self.assertTrue(Path(task.command[1]).is_file(), task.command[1])
                if task.kind == 'train':
                    self.assertIn('--save-last', task.command)
                    self.assertEqual(task.command[task.command.index('--epochs') + 1], '60')
                    self.assertEqual(task.command[task.command.index('--batch-size') + 1], '64')
            if classes == 1000:
                self.assertTrue(manifest['data_root'].endswith('/MSASL1000'))

    def test_invalid_variants_rejected(self):
        with self.assertRaises(ValueError):
            self.plan('wlasl', 200)
        with self.assertRaises(ValueError):
            self.plan('msasl', 300)

    def test_generated_training_arguments_parse(self):
        # Import each script with its dataset-local finetune adapter, like a direct launch.
        import importlib.util
        import sys
        from unittest.mock import patch
        for family in ('wlasl', 'msasl'):
            tasks, _ = self.plan(family, 300 if family == 'wlasl' else 1000)
            for task in tasks:
                if task.kind != 'train':
                    continue
                previous = sys.modules.pop('finetune', None)
                sys.path.insert(0, str(ROOT / 'training' / family))
                try:
                    spec = importlib.util.spec_from_file_location(task.name, task.command[1])
                    module = importlib.util.module_from_spec(spec)
                    # dataclass resolves its module through sys.modules.
                    sys.modules[task.name] = module
                    spec.loader.exec_module(module)
                    with patch.object(sys, 'argv', task.command[1:]):
                        args = module.parse_args()
                    self.assertEqual(args.num_class, 300 if family == 'wlasl' else 1000)
                finally:
                    sys.path.pop(0)
                    sys.modules.pop('finetune', None)
                    sys.modules.pop(task.name, None)
                    if previous is not None:
                        sys.modules['finetune'] = previous

    def test_generated_evaluation_and_quantization_arguments_parse(self):
        import importlib.util
        import sys
        from unittest.mock import patch
        sys.path.insert(0, str(ROOT / 'analysis'))
        try:
            for family in ('wlasl', 'msasl'):
                tasks, _ = self.plan(family, 300 if family == 'wlasl' else 1000)
                for task in tasks:
                    if task.kind not in ('evaluate', 'quantize'):
                        continue
                    name = family + '_' + task.name
                    spec = importlib.util.spec_from_file_location(name, task.command[1])
                    module = importlib.util.module_from_spec(spec)
                    sys.modules[name] = module
                    try:
                        spec.loader.exec_module(module)
                        with patch.object(sys, 'argv', task.command[1:]):
                            args = module.parse_args()
                        self.assertEqual(args.batch_size, 32)
                        self.assertEqual(args.dropout, 0.0)
                    finally:
                        sys.modules.pop(name, None)
        finally:
            sys.path.pop(0)


class RestartTests(unittest.TestCase):
    def test_optimizer_scheduler_rng_and_best_restore(self):
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp)
            args = SimpleNamespace(save_last=True, resume_stage=False, out_dir=str(directory), lr=0.01)
            model = torch.nn.Linear(4, 2)
            optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
            scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, [1], gamma=0.1)
            model(torch.ones(2, 4)).sum().backward()
            optimizer.step()
            scheduler.step()
            best = {'state_dict': model.state_dict(), 'epoch': 1}
            torch.save(best, directory / 'best.pth.tar')
            stage = StageState(args, model, optimizer, scheduler, directory)
            stage.save(1, 0.8, 1, [{'epoch': 1}], 0)
            expected_rng = (random.random(), np.random.rand(), torch.rand(4))
            expected_state = {key: tensor.clone() for key, tensor in model.state_dict().items()}
            # A future partial epoch must not replace the saved validation-best on resume.
            torch.save({'epoch': 99}, directory / 'best.pth.tar')
            args.resume_stage = True
            resumed = torch.nn.Linear(4, 2)
            new_optimizer = torch.optim.SGD(resumed.parameters(), lr=0.01, momentum=0.9)
            new_scheduler = torch.optim.lr_scheduler.MultiStepLR(new_optimizer, [1], gamma=0.1)
            state = StageState(args, resumed, new_optimizer, new_scheduler, directory).restore()
            self.assertEqual(state, (2, 0.8, 1, [{'epoch': 1}], 0))
            self.assertEqual(new_optimizer.param_groups[0]['lr'], 0.001)
            self.assertEqual(len(new_optimizer.state), 2)
            self.assertEqual(new_scheduler.state_dict(), scheduler.state_dict())
            self.assertEqual(random.random(), expected_rng[0])
            self.assertEqual(np.random.rand(), expected_rng[1])
            self.assertTrue(torch.equal(torch.rand(4), expected_rng[2]))
            for key, tensor in resumed.state_dict().items():
                self.assertTrue(torch.equal(tensor, expected_state[key]))
            self.assertEqual(trusted_load(directory / 'best.pth.tar')['epoch'], 1)
            args.lr = 0.02
            with self.assertRaises(ValueError):
                StageState(args, resumed, new_optimizer, new_scheduler, directory).restore()

    def test_legacy_launch_does_not_write_restart_state(self):
        with tempfile.TemporaryDirectory() as tmp:
            model = torch.nn.Linear(2, 2)
            stage = StageState(SimpleNamespace(), model, torch.optim.SGD(model.parameters(), lr=0.01), None, tmp)
            self.assertEqual(stage.restore(), (1, -1.0, -1, [], 0))
            stage.save(1, 0.8, 1, [], 0)
            self.assertFalse((Path(tmp) / 'last.pth.tar').exists())


class ExportTests(unittest.TestCase):
    def test_deltas_and_standardized_metrics_not_report_latency(self):
        baseline = dict(accuracy=0.8, parameter_count=100, latency_ms_per_batch=20,
                        model_size_mb=100, gflops_per_sample=1)
        compressed = dict(baseline, accuracy=0.75, model_size_mb=50)
        self.assertAlmostEqual(deltas(compressed, baseline)['accuracy_pp'], -5)
        self.assertEqual(deltas(compressed, baseline)['model_size_mb_relative_percent'], -50)
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _, manifest = build_plan(ROOT, '/python', '/workspace', 'wlasl', 100, root)
            benchmarks = {}
            for entry in manifest['models']:
                checkpoint = Path(entry['checkpoint'])
                checkpoint.parent.mkdir(parents=True, exist_ok=True)
                checkpoint.write_bytes(b'fixture')
                if entry['quantized']:
                    path = Path(entry['quantized_checkpoint'])
                    path.parent.mkdir(parents=True, exist_ok=True)
                    path.write_bytes(b'quantized fixture')
                source = entry.get('quantized_checkpoint', entry['checkpoint'])
                metrics = dict(baseline, latency_ms_per_batch=999)
                write_json(entry['report'], dict(checkpoint=str(checkpoint), metrics=metrics,
                                                after_int8_supported_fp16_fallback=metrics))
                write_json(entry['predictions'], dict(records=[dict(index=i, clip=str(i), target=0,
                                                                    prediction=0 if i < 4 else 1) for i in range(5)]))
                benchmarks[entry['id']] = dict(baseline, checkpoint_sha256=sha256(source),
                                               precision='FP32', p25_ms=19, p75_ms=21, batch_size=32,
                                               flop_audit=dict(unsupported_operators={}, by_operator={'aten::linear': 32000000000}))
            write_json(root / 'models.json', manifest)
            write_json(root / 'benchmark.json', dict(dataset='wlasl100', models=benchmarks))
            export(root / 'models.json', root / 'benchmark.json', root / 'comparisons')
            result = json.loads((root / 'comparisons/results.json').read_text())
            self.assertEqual(len(result['results']), 14)
            self.assertEqual(result['results'][0]['metrics']['latency_ms_per_batch'], 20)
            self.assertEqual(result['results'][0]['metrics']['parameter_count'], 100)
            self.assertEqual(result['prediction_comparisons'][0]['prediction_disagreements'], 0)
            prediction_path = manifest['models'][1]['predictions']
            original = read_json(prediction_path)
            changed = json.loads(json.dumps(original))
            changed['records'][0]['prediction'] = 1
            changed['records'][4]['prediction'] = 0
            write_json(prediction_path, changed)
            export(root / 'models.json', root / 'benchmark.json', root / 'different_predictions')
            comparison = read_json(root / 'different_predictions/results.json')['prediction_comparisons'][0]
            self.assertTrue(comparison['same_accuracy'])
            self.assertEqual(comparison['prediction_disagreements'], 2)
            write_json(prediction_path, original)
            # Export now runs through the shared runner's internal CLI mode.
            import subprocess
            import sys
            subprocess.run([
                sys.executable, str(ROOT / 'experiments/run_study.py'),
                '--export-manifest', str(root / 'models.json'),
                '--benchmark', str(root / 'benchmark.json'),
                '--out-dir', str(root / 'cli_comparisons'),
            ], check=True, stdout=subprocess.DEVNULL)
            self.assertEqual(json.loads((root / 'cli_comparisons/results.json').read_text()), result)
            prediction_path = manifest['models'][1]['predictions']
            original = read_json(prediction_path)
            changed = json.loads(json.dumps(original))
            changed['records'][0]['prediction'] = 1
            write_json(prediction_path, changed)
            with self.assertRaisesRegex(ValueError, 'reproduce reported accuracy'):
                export(root / 'models.json', root / 'benchmark.json', root / 'comparisons')
            write_json(prediction_path, original)
            Path(manifest['models'][0]['checkpoint']).write_bytes(b'changed')
            with self.assertRaises(ValueError):
                export(root / 'models.json', root / 'benchmark.json', root / 'comparisons')

    def test_weight_audit_ignores_packaging_and_parameter_count_survives_quantization(self):
        from analysis.benchmark_study import audit_weights, parameter_count
        from analysis.quantization_report import replace_int8_supported_layers
        model = torch.nn.Sequential(torch.nn.Linear(4, 2), torch.nn.BatchNorm1d(2))
        count = parameter_count(model)
        self.assertEqual(count, 14)
        replace_int8_supported_layers(model)
        self.assertEqual(parameter_count(model), count)
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            torch.save(dict(state_dict=model.state_dict(), epoch=1), root / 'a.pth')
            torch.save(dict(state_dict=model.state_dict(), epoch=60), root / 'b.pth')
            paths = dict(first=str(root / 'a.pth'), second=str(root / 'b.pth'))
            self.assertNotEqual(sha256(paths['first']), sha256(paths['second']))
            with self.assertRaises(ValueError):
                audit_weights(paths, root / 'audit.json')
            self.assertEqual(read_json(root / 'audit.json')['duplicate_groups'], [['first', 'second']])
            model[0].qweight[0, 0] += 1
            torch.save(dict(state_dict=model.state_dict()), root / 'b.pth')
            audit_weights(paths, root / 'audit.json')
            self.assertEqual(read_json(root / 'audit.json')['duplicate_groups'], [])

    def test_actual_evaluations_record_predictions_without_changing_accuracy(self):
        import importlib
        from unittest.mock import patch
        from torch.utils.data import Dataset, DataLoader

        class Clips(Dataset):
            base_ds = SimpleNamespace(video_list=[['clip-a', '0'], ['clip-b', '1']])

            def __len__(self):
                return 2

            def __getitem__(self, index):
                return {'values': torch.tensor([2., 0.] if index == 0 else [0., 2.])}, index

        class Model(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.anchor = torch.nn.Parameter(torch.zeros(1))

            def forward(self, x):
                return x['values'] + self.anchor

        with tempfile.TemporaryDirectory() as tmp:
            for family in ('', 'msasl_'):
                module = importlib.import_module('analysis.' + family + 'quantization_report')
                path = Path(tmp) / (family + 'state.pth')
                loader = DataLoader(Clips(), batch_size=2, shuffle=False)
                with patch.dict(os.environ, {'MASA_STUDY_PREDICTIONS': '1'}):
                    result = module.evaluate_model(Model(), loader, torch.nn.CrossEntropyLoss(),
                                                   torch.device('cpu'), path, 0)
                records = read_json(path.with_suffix('.predictions.json'))
                self.assertEqual(result.acc, 1)
                self.assertEqual(records['correct'], 2)
                self.assertEqual([row['clip'] for row in records['records']], ['clip-a', 'clip-b'])

    def test_quantized_benchmark_restores_saved_hybrid_weights(self):
        from unittest.mock import patch
        from analysis.benchmark_study import load_model
        from analysis.quantization_report import replace_int8_supported_layers

        class SmallModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.fc = torch.nn.Linear(4, 2)

            def forward(self, x):
                return self.fc(x)

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            model = SmallModel()
            torch.save({'state_dict': model.state_dict()}, root / 'fp32.pth')
            replace_int8_supported_layers(model)
            model.half()
            # A saved hybrid state must be restored, rather than silently regenerated.
            model.fc.qweight.fill_(7)
            torch.save(model.state_dict(), root / 'hybrid.pth')
            entry = dict(architecture='baseline', quantized=True,
                         checkpoint=str(root / 'fp32.pth'), quantized_checkpoint=str(root / 'hybrid.pth'))
            with patch('analysis.benchmark_study.MASA', side_effect=lambda **kwargs: SmallModel()):
                restored = load_model(entry, 2)
            for key, value in model.state_dict().items():
                self.assertTrue(torch.equal(value, restored.state_dict()[key]), key)
            self.assertEqual(restored.fc.qweight.dtype, torch.int8)


class FlopTests(unittest.TestCase):
    def test_linear_grouped_convolution_and_lowrank_formulas(self):
        from analysis.benchmark_study import count_forward_flops
        from moco.low_rank_modules import LowRankLinear
        dense = torch.nn.Linear(4, 2).eval()
        x = torch.randn(3, 4)
        self.assertEqual(count_forward_flops(dense, x)['flops'], 54)
        lowrank = LowRankLinear(4, 2, 1).eval()
        self.assertEqual(count_forward_flops(lowrank, x)['flops'], 42)
        conv = torch.nn.Conv2d(4, 6, 3, groups=2).eval()
        self.assertEqual(count_forward_flops(conv, torch.randn(2, 4, 5, 5))['flops'], 3996)

    def test_attention_and_gcn_contractions(self):
        from analysis.benchmark_study import count_forward_flops

        class Attention(torch.nn.Module):
            def forward(self, x):
                scores = x @ x.transpose(-1, -2)
                return (scores / 2).softmax(-1) @ x

        result = count_forward_flops(Attention().eval(), torch.randn(2, 3, 5, 4))
        self.assertEqual(result['by_operator']['aten::matmul'], 2400)
        self.assertEqual(result['flops'], 3120)

        class GCN(torch.nn.Module):
            def forward(self, x):
                return torch.einsum('nkctv,kvw->nctw', x['features'], x['adjacency'])

        result = count_forward_flops(GCN().eval(),
                                     {'features': torch.randn(2, 3, 4, 5, 6), 'adjacency': torch.randn(3, 6, 7)})
        self.assertEqual(result['flops'], 10080)

    def test_normalization_formulas(self):
        from analysis.benchmark_study import count_forward_flops
        from moco.CMABert_backbone import LayerNorm
        self.assertEqual(count_forward_flops(LayerNorm(4).eval(), torch.randn(2, 4))['flops'], 68)
        self.assertEqual(count_forward_flops(torch.nn.BatchNorm1d(2).eval(), torch.randn(3, 2))['flops'], 28)

    def test_addmm_matches_linear_and_counts_scaling(self):
        from analysis.benchmark_study import count_forward_flops

        class AddMM(torch.nn.Module):
            def __init__(self, alpha=1, beta=1):
                super().__init__()
                self.weight = torch.nn.Parameter(torch.randn(4, 2))
                self.bias = torch.nn.Parameter(torch.randn(2))
                self.alpha, self.beta = alpha, beta

            def forward(self, x):
                return torch.addmm(self.bias, x, self.weight, beta=self.beta, alpha=self.alpha)

        x = torch.randn(3, 4)
        for alpha, beta, expected in [(1, 1, 54), (1, 0, 48), (2, 1, 60), (2, 3, 66)]:
            result = count_forward_flops(AddMM(alpha, beta).eval(), x)
            self.assertEqual(result['by_operator'], {'aten::addmm': expected})
        self.assertEqual(count_forward_flops(torch.nn.Linear(4, 2).eval(), x)['flops'], 54)

    def test_quantization_overhead_unknown_ops_and_control_flow(self):
        from analysis.benchmark_study import count_forward_flops
        from analysis.quantization_report import replace_int8_supported_layers
        model = torch.nn.Sequential(torch.nn.Linear(4, 2)).eval()
        replace_int8_supported_layers(model)
        model.eval()
        result = count_forward_flops(model, torch.randn(3, 4))
        self.assertEqual(result['flops'], 54 + 8)

        class Unsupported(torch.nn.Module):
            def forward(self, x):
                return x.sin()

        with self.assertRaisesRegex(ValueError, 'Unsupported FLOP operators'):
            count_forward_flops(Unsupported().eval(), torch.randn(2, 3))

        class Conditional(torch.nn.Module):
            def forward(self, x):
                if x.sum() > 0:
                    return x * 2
                return x + 1

        with self.assertRaisesRegex(ValueError, 'Unaudited tracing warning'):
            count_forward_flops(Conditional().eval(), torch.ones(2, 3))

    def test_real_masa_all_layouts_have_audited_coverage(self):
        from analysis.benchmark_study import preflight_flop_layouts
        threads = torch.get_num_threads()
        try:
            torch.set_num_threads(2)
            inputs = dict(rh=torch.randn(1, 32, 21, 2), lh=torch.randn(1, 32, 21, 2),
                          body=torch.randn(1, 32, 7, 2), mask=torch.zeros(1, 64, 21, 2))
            audits = preflight_flop_layouts(inputs, 100, torch.device('cpu'))
            self.assertEqual(len(audits), 6)
            for result in audits.values():
                self.assertEqual(result['unsupported_operators'], {})
                self.assertGreater(result['by_operator']['aten::matmul'], 0)
                self.assertGreater(result['by_operator']['aten::einsum'], 0)
            self.assertLess(audits['ghost_fp32']['flops'], audits['baseline_fp32']['flops'])
            self.assertLess(audits['lowrank_fp32']['flops'], audits['baseline_fp32']['flops'])
        finally:
            torch.set_num_threads(threads)


class RelocationTests(unittest.TestCase):
    def test_preview_and_apply_preserve_relative_links(self):
        from experiments.run_study import repair
        with tempfile.TemporaryDirectory() as tmp:
            workspace = Path(tmp)
            wlasl = workspace / 'datasets/WLASL/latest_full_wlasl'
            msasl = workspace / 'datasets/MSASL/native_masa_ready'
            wlasl.mkdir(parents=True)
            (msasl / 'MSASL').mkdir(parents=True)
            (msasl / 'MSASL1000').mkdir()
            target = msasl / 'MSASL/pose.pkl'
            target.write_bytes(b'pose')
            absolute = msasl / 'MSASL1000/pose.pkl'
            absolute.symlink_to('/workspace/MSASL/native_masa_ready/MSASL/pose.pkl')
            relative = wlasl / 'relative'
            relative.symlink_to('../../MSASL/native_masa_ready/MSASL/pose.pkl')
            repair(workspace)
            self.assertEqual(os.readlink(absolute), '/workspace/MSASL/native_masa_ready/MSASL/pose.pkl')
            repair(workspace, apply=True)
            self.assertEqual(absolute.read_bytes(), b'pose')
            self.assertEqual(os.readlink(relative), '../../MSASL/native_masa_ready/MSASL/pose.pkl')


class OrchestrationTests(unittest.TestCase):
    def test_single_launch_and_verified_resume(self):
        import sys
        from unittest.mock import patch
        from experiments.run_study import run

        with tempfile.TemporaryDirectory() as tmp:
            workspace = Path(tmp).resolve()
            pretrain = workspace / 'checkpoints/pretrained_model.pth.tar'
            pretrain.parent.mkdir()
            pretrain.write_bytes(b'pretrain')
            args = SimpleNamespace(family='wlasl', classes=100, workspace=workspace,
                                   results_root=workspace / 'results/study', workers=0,
                                   benchmark_warmup=1, benchmark_repetitions=2, benchmark_rounds=1,
                                   dry_run=False, resume=False)
            output = args.results_root / 'wlasl100'
            tasks, _ = build_plan(ROOT, sys.executable, workspace, 'wlasl', 100, output, 0, 1, 2, 1)
            executed = []
            provenance = dict(gpu_uuid='GPU-test', splits={}, metadata_sha256={}, pretrained_sha256='hash',
                              packages={}, cuda='test', gpu='test', python='test', driver='test', cudnn=1,
                              training_backend_defaults={})

            def fake_execute(command, log, env):
                if '--preflight-manifest' in command:
                    write_json(command[command.index('--out') + 1], provenance)
                    return
                task = next(item for item in tasks if item.command == command)
                executed.append(task.name)
                for path in task.outputs:
                    Path(path).parent.mkdir(parents=True, exist_ok=True)
                    Path(path).write_bytes(b'fixture')
                if task.kind == 'train':
                    write_json(task.outputs[1], dict(best_epoch=60))
                    write_json(task.outputs[2], [dict(epoch=i, val_acc=i / 100) for i in range(1, 61)])
                elif task.kind in ('evaluate', 'quantize'):
                    metrics = dict(accuracy=0.8)
                    write_json(task.outputs[0], dict(metrics=metrics, after_int8_supported_fp16_fallback=metrics))

            with patch('experiments.run_study.execute', side_effect=fake_execute), \
                    patch('experiments.run_study.gpu_uuid', return_value='GPU-test'), \
                    patch('experiments.run_study.source_record', return_value=dict(source_sha256={'file': 'hash'})), \
                    patch.dict('os.environ', {'SLURM_CPUS_PER_TASK': '4'}):
                run(args)
                self.assertEqual(executed, [task.name for task in tasks])
                self.assertEqual(len(list((output / 'task_records').glob('*.json'))), 25)
                args.resume = True
                run(args)
                self.assertEqual(len(executed), 25)
                Path(tasks[0].outputs[0]).write_bytes(b'changed checkpoint')
                with self.assertRaises(ValueError):
                    run(args)


if __name__ == '__main__':
    unittest.main()
