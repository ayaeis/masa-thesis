"""CPU tests for archive selection and two-stage low-rank checkpoint loading."""

import ast
import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import torch
from torch import nn

from data_prep.msasl import archive_loader
from feeder.single_dataset.MSASL import MSASL
from moco.low_rank_modules import apply_low_rank_from_dense, apply_low_rank_structure


ROOT = Path(__file__).resolve().parents[1]


def import_trainer(family):
    directory = ROOT / 'training' / family
    previous = sys.modules.pop('finetune', None)
    sys.path.insert(0, str(directory))
    try:
        spec = importlib.util.spec_from_file_location(
            family + '_continuation', directory / 'finetune_lowrank_architecture_first.py'
        )
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module
    finally:
        sys.path.pop(0)
        sys.modules.pop('finetune', None)
        if previous is not None:
            sys.modules['finetune'] = previous


class TinyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder_q = nn.Module()
        self.encoder_q.GCN_Tran = nn.Sequential(nn.Linear(8, 8))

    def forward(self, x):
        return self.encoder_q.GCN_Tran(x)


class ArchiveTests(unittest.TestCase):
    def test_selection_and_argument_forwarding(self):
        for classes, selected in [(100, 'MSASLArchive'), (200, 'MSASLArchive'),
                                  (1000, 'MSASL1000Archive')]:
            with patch.object(archive_loader, 'MSASLArchive') as old, \
                    patch.object(archive_loader, 'MSASL1000Archive') as full:
                archive_loader.load_archive('/dataset', 'test', classes, False, threshold=0.5)
                chosen = full if selected == 'MSASL1000Archive' else old
                other = old if chosen is full else full
                chosen.assert_called_once_with(
                    data_root='/dataset', data_split='test', class_num=classes,
                    use_cache=False, threshold=0.5,
                )
                other.assert_not_called()

    def test_msasl1000_excludes_train_val_and_test_records(self):
        def fake_base(instance, data_root, data_split, **kwargs):
            if data_split == 'train':
                instance.flag = 2
                instance.video_list = np.array([
                    ['train_keep.mp4', '0'], ['train_bad.mp4', '1'],
                    ['val_keep.mp4', '2'], ['val_bad.mp4', '3'],
                ])
            else:
                instance.video_list = np.array([
                    ['test_keep.mp4', '0'], ['test_bad.mp4', '1'],
                ])

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / 'unavailable_records.json').write_text(json.dumps([
                dict(split='train', filename='train_bad.mp4')
            ]))
            (root / 'native_feeder_exclusions.json').write_text(json.dumps([
                dict(split='val', filename='val_bad.mp4'),
                dict(split='test', filename='test_bad.mp4'),
            ]))
            # There is deliberately no MSASL200 metadata file in this fixture.
            with patch.object(MSASL, '__init__', fake_base):
                train = archive_loader.load_archive(root, 'train', 1000)
                test = archive_loader.load_archive(root, 'test', 1000)
            self.assertEqual(train.flag, 1)
            self.assertEqual(train.video_list[:, 0].tolist(), ['train_keep.mp4', 'val_keep.mp4'])
            self.assertEqual(test.video_list[:, 0].tolist(), ['test_keep.mp4'])

    def test_all_msasl_entrypoints_use_selector(self):
        paths = list((ROOT / 'training/msasl').glob('finetune*.py')) + [
            ROOT / 'analysis/msasl_checkpoint_metrics.py',
            ROOT / 'analysis/msasl_quantization_report.py',
        ]
        for path in paths:
            imports = [node for node in ast.parse(path.read_text()).body
                       if isinstance(node, ast.ImportFrom)]
            self.assertTrue(any(node.module == 'data_prep.msasl.archive_loader'
                                and any(alias.name == 'load_archive' and alias.asname == 'MSASL'
                                        for alias in node.names) for node in imports), path)


class ContinuationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.wlasl = import_trainer('wlasl')
        cls.msasl = import_trainer('msasl')

    def test_training_logic_matches_msasl(self):
        wlasl = (ROOT / 'training/wlasl/finetune_lowrank_architecture_first.py').read_text()
        msasl = (ROOT / 'training/msasl/finetune_lowrank_architecture_first.py').read_text()
        normalized = wlasl.replace(
            'from feeder.single_dataset.WLASL import WLASL',
            'from data_prep.msasl.archive_loader import load_archive as MSASL',
        ).replace('WLASLSupervised', 'MSASLSupervised').replace(
            '    p.add_argument("--subset-num", type=int, default=100)\n', ''
        ).replace('base_train = WLASL(', 'base_train = MSASL(').replace(
            'base_test = WLASL(', 'base_test = MSASL('
        ).replace('subset_num=args.subset_num', 'class_num=args.num_class')
        self.assertEqual(ast.dump(ast.parse(normalized)), ast.dump(ast.parse(msasl)))

    def test_restore_factors_exactly_and_reject_bad_checkpoints(self):
        args = SimpleNamespace(rank_ratio=0.125, low_rank_targets='transformer',
                               low_rank_min_features=1)
        model = TinyModel()
        apply_low_rank_from_dense(model, 0.125, 'transformer', 1)
        # Simulate learned factors that differ from the SVD initialization.
        with torch.no_grad():
            for parameter in model.parameters():
                parameter.add_(0.25)
        checkpoint = dict(state_dict=model.state_dict(), rank_ratio=0.125,
                          low_rank_targets='transformer', low_rank_min_features=1)
        inputs = torch.randn(2, 8)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'stage1.pth.tar'
            torch.save(checkpoint, path)
            for trainer in (self.wlasl, self.msasl):
                restored = TinyModel()
                with patch('torch.linalg.svd', side_effect=AssertionError('Stage 2 must not run SVD')):
                    apply_low_rank_structure(restored, 0.125, 'transformer', 1)
                    trainer.validate_resume_configuration(checkpoint, args)
                    trainer.load_low_rank_checkpoint_strict(restored, path)
                for key, value in model.state_dict().items():
                    self.assertTrue(torch.equal(value, restored.state_dict()[key]), key)
                self.assertTrue(torch.equal(model(inputs), restored(inputs)))
                with self.assertRaises(ValueError):
                    trainer.validate_resume_configuration(dict(checkpoint, rank_ratio=0.25), args)
                broken = dict(checkpoint, state_dict=dict(checkpoint['state_dict']))
                broken['state_dict'].pop(next(iter(broken['state_dict'])))
                torch.save(broken, path)
                with self.assertRaises(RuntimeError):
                    trainer.load_low_rank_checkpoint_strict(restored, path)
                torch.save(checkpoint, path)

    def test_new_stage_resets_optimizer_and_schedule(self):
        trainer = self.wlasl
        initial_model = TinyModel()
        apply_low_rank_from_dense(initial_model, 0.125, 'transformer', 1)
        args = SimpleNamespace(
            seed=123, data_root='unused', subset_num=100, num_class=100,
            target_t=32, temporal_sampling='index', batch_size=2, workers=0,
            dropout=0.0, init_mode='resume_low_rank', rank_ratio=0.125,
            low_rank_targets='transformer', low_rank_min_features=1,
            train_temporal_crop_min=1.0, warmup_epochs=0, head_lr_mult=1.0,
            optim='sgd', lr=0.01, momentum=0.9, weight_decay=0.0,
            scheduler='multistep', milestones=[1], lr_gamma=0.1,
            freeze_epochs=0, epochs=1, min_delta=0.0, patience=999,
        )
        seen = []

        def fake_epoch(model, loader, criterion, optimizer, device, train):
            if train:
                seen.append((optimizer.param_groups[0]['lr'], len(optimizer.state)))
                optimizer.zero_grad()
                model(torch.ones(2, 8)).sum().backward()
                optimizer.step()
            return SimpleNamespace(loss=0.5, acc=0.75)

        class FakeBase:
            flag = 1

            def __len__(self):
                return 2

        with tempfile.TemporaryDirectory() as tmp:
            args.init_ckpt = str(Path(tmp) / 'stage1.pth.tar')
            args.out_dir = str(Path(tmp) / 'stage2')
            # An old optimizer state is present but must not be reused.
            torch.save(dict(state_dict=initial_model.state_dict(), optimizer={'old': True},
                            rank_ratio=0.125, low_rank_targets='transformer',
                            low_rank_min_features=1), args.init_ckpt)
            with patch.object(trainer, 'parse_args', return_value=args), \
                    patch.object(trainer, 'WLASL', return_value=FakeBase()), \
                    patch.object(trainer, 'WLASLSupervised', return_value=[]), \
                    patch.object(trainer, 'DataLoader', return_value=[]), \
                    patch.object(trainer, 'MASA', side_effect=lambda **kwargs: TinyModel()), \
                    patch.object(trainer, 'run_epoch', side_effect=fake_epoch), \
                    patch.object(trainer.torch.cuda, 'is_available', return_value=False), \
                    patch.object(trainer, 'apply_low_rank_from_dense', side_effect=AssertionError('Refactorization')), \
                    patch.object(trainer, 'Subset', return_value=[]):
                trainer.main()
            self.assertEqual(seen, [(0.01, 0)])
            history = json.loads((Path(args.out_dir) / 'metrics.json').read_text())
            self.assertAlmostEqual(history[0]['lr'], 0.001)
            summary = json.loads((Path(args.out_dir) / 'summary.json').read_text())
            self.assertEqual(summary['init_mode'], 'resume_low_rank')


if __name__ == '__main__':
    unittest.main()
