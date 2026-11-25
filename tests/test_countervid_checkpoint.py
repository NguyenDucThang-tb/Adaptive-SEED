import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from seed_video.training.countervid_checkpoint import load_checkpoint, save_checkpoint


def test_restore_optimizer_rng_and_config(tmp_path):
    param = torch.nn.Parameter(torch.tensor([1.0]))
    optimizer = torch.optim.AdamW([param], lr=0.01)
    (param ** 2).sum().backward()
    optimizer.step()
    saved = param.detach().clone()
    torch.manual_seed(7)
    config = {'max_steps': 2, 'method': 'adaptive'}
    path = tmp_path / 'checkpoint.pt'
    save_checkpoint(path, config=config, step=1, adapter={'param': saved}, optimizer=optimizer)
    expected_random = torch.rand(2)
    optimizer.state.clear()
    param.data.zero_()
    torch.manual_seed(1)

    def restore(adapter):
        param.data.copy_(adapter['param'])

    assert load_checkpoint(path, config=config, optimizer=optimizer, set_adapter=restore) == 1
    assert torch.equal(param.detach(), saved)
    assert torch.equal(torch.rand(2), expected_random)
    assert optimizer.state[param]['step'].item() == 1
    with pytest.raises(ValueError, match='configuration'):
        load_checkpoint(path, config={'max_steps': 3}, optimizer=optimizer, set_adapter=restore)


def test_resumed_trajectory_matches_uninterrupted(tmp_path):
    def update(param, optimizer):
        optimizer.zero_grad()
        loss = ((param * torch.rand(3)) ** 2).sum()
        loss.backward()
        optimizer.step()

    config = {'max_steps': 3, 'method': 'adaptive'}
    torch.manual_seed(23)
    param = torch.nn.Parameter(torch.ones(3))
    optimizer = torch.optim.AdamW([param], lr=0.01)
    update(param, optimizer)
    path = tmp_path / 'checkpoint.pt'
    save_checkpoint(path, config=config, step=1, adapter={'param': param.detach().clone()},
                    optimizer=optimizer)
    update(param, optimizer)
    update(param, optimizer)
    expected = param.detach().clone()

    resumed = torch.nn.Parameter(torch.zeros(3))
    resumed_optimizer = torch.optim.AdamW([resumed], lr=0.01)
    load_checkpoint(path, config=config, optimizer=resumed_optimizer,
                    set_adapter=lambda data: resumed.data.copy_(data['param']))
    update(resumed, resumed_optimizer)
    update(resumed, resumed_optimizer)
    assert torch.equal(resumed.detach(), expected)
