"""Atomic, configuration-checked CounterVid training checkpoints."""
import os
from pathlib import Path

import torch


def save_checkpoint(path, *, config, step, adapter, optimizer):
    path = Path(path)
    state = {'version': 1, 'config': config, 'step': step, 'adapter': adapter,
             'optimizer': optimizer.state_dict(), 'cpu_rng': torch.get_rng_state(),
             'cuda_rng': torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None}
    temporary = path.with_name(path.name + f'.tmp.{os.getpid()}')
    try:
        torch.save(state, temporary)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def load_checkpoint(path, *, config, optimizer, set_adapter):
    state = torch.load(path, map_location='cpu', weights_only=False)
    if state.get('version') != 1 or state.get('config') != config:
        raise ValueError('Checkpoint configuration does not match this run')
    if not isinstance(state.get('step'), int) or not 0 <= state['step'] <= config['max_steps']:
        raise ValueError('Invalid checkpoint step')
    set_adapter(state['adapter'])
    optimizer.load_state_dict(state['optimizer'])
    torch.set_rng_state(state['cpu_rng'])
    if state['cuda_rng'] is not None:
        if not torch.cuda.is_available() or len(state['cuda_rng']) != torch.cuda.device_count():
            raise ValueError('Checkpoint CUDA device count differs from this run')
        torch.cuda.set_rng_state_all(state['cuda_rng'])
    return state['step']
