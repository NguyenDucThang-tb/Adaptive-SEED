import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from seed_video.training.countervid_loss import adaptive_sequence_loss


def test_gain_mask_and_gradient_isolation():
    student = torch.zeros(1, 3, 3, requires_grad=True)
    teacher = torch.tensor([[[3., 0., 0.], [0., 3., 0.], [99., 0., 0.]]], requires_grad=True)
    base = torch.tensor([[[0., 3., 0.], [3., 0., 0.], [0., 99., 0.]]], requires_grad=True)
    labels = torch.tensor([[0, 0, -100]])
    mask = torch.tensor([[1, 1, 0]])
    loss, stats = adaptive_sequence_loss(student, teacher, base, labels, mask)
    loss.backward()
    assert stats['answer_tokens'].item() == 2
    assert stats['mean_weight'].item() == pytest.approx(0.5)
    assert student.grad[0, :2].abs().sum() > 0
    assert student.grad[0, 2].abs().sum() == 0
    assert teacher.grad is None and base.grad is None


def test_empty_mask_and_invalid_token_rejected():
    z = torch.zeros(1, 1, 2)
    with pytest.raises(ValueError, match='at least one'):
        adaptive_sequence_loss(z, z, z, torch.zeros(1, 1), torch.zeros(1, 1))
    with pytest.raises(ValueError, match='vocabulary'):
        adaptive_sequence_loss(z, z, z, torch.full((1, 1), 2), torch.ones(1, 1))
