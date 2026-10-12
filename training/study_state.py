"""Optional epoch-boundary restart state; legacy launches remain unchanged."""

import inspect
import os
import random
from pathlib import Path

import numpy as np
import torch


def trusted_load(path):
    kwargs = {'map_location': 'cpu'}
    if 'weights_only' in inspect.signature(torch.load).parameters:
        kwargs['weights_only'] = False
    return torch.load(path, **kwargs)


def atomic_torch_save(value, path):
    path = Path(path)
    temporary = path.with_name(path.name + '.tmp')
    torch.save(value, temporary)
    os.replace(temporary, path)


def add_restart_args(parser):
    parser.add_argument('--save-last', action='store_true',
                        help='Save complete epoch-boundary restart state.')
    parser.add_argument('--resume-stage', action='store_true',
                        help='Restore last.pth.tar within this stage, including optimizer/scheduler/RNG.')


class StageState:
    def __init__(self, args, model, optimizer, scheduler, out_dir):
        self.args = args
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.directory = Path(out_dir)
        self.path = self.directory / 'last.pth.tar'

    def configuration(self):
        return {key: value for key, value in vars(self.args).items()
                if key not in ('save_last', 'resume_stage')}

    def restore(self):
        if not getattr(self.args, 'resume_stage', False):
            return 1, -1.0, -1, [], 0
        checkpoint = trusted_load(self.path)
        if checkpoint['configuration'] != self.configuration():
            raise ValueError('Restart configuration differs from the saved stage.')
        self.model.load_state_dict(checkpoint['state_dict'], strict=True)
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        if self.scheduler is not None:
            self.scheduler.load_state_dict(checkpoint['scheduler'])
        for name, parameter in self.model.named_parameters():
            parameter.requires_grad = checkpoint['requires_grad'][name]
        # Restore the matching best snapshot too, even if interruption happened
        # between a newer best-file write and the last-state commit.
        atomic_torch_save(checkpoint['best_checkpoint'], self.directory / 'best.pth.tar')
        random.setstate(checkpoint['python_rng'])
        np.random.set_state(checkpoint['numpy_rng'])
        torch.set_rng_state(checkpoint['torch_rng'])
        if torch.cuda.is_available():
            torch.cuda.set_rng_state_all(checkpoint['cuda_rng'])
        print('[resume] continuing after epoch {}'.format(checkpoint['epoch']), flush=True)
        return (checkpoint['epoch'] + 1, checkpoint['best_val'],
                checkpoint['best_epoch'], checkpoint['history'], checkpoint['no_improve'])

    def save(self, epoch, best_val, best_epoch, history, no_improve):
        if not getattr(self.args, 'save_last', False):
            return
        checkpoint = {
            'epoch': epoch,
            'state_dict': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'scheduler': self.scheduler.state_dict() if self.scheduler is not None else None,
            'configuration': self.configuration(),
            'requires_grad': {name: value.requires_grad for name, value in self.model.named_parameters()},
            'best_val': best_val, 'best_epoch': best_epoch,
            'history': history, 'no_improve': no_improve,
            'best_checkpoint': trusted_load(self.directory / 'best.pth.tar'),
            'python_rng': random.getstate(), 'numpy_rng': np.random.get_state(),
            'torch_rng': torch.get_rng_state(),
            'cuda_rng': torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
        }
        atomic_torch_save(checkpoint, self.path)
