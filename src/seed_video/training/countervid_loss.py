"""Token-level gain-weighted KD for the 16-frame/8-frame CounterVid variant."""
import math

import torch
import torch.nn.functional as F


def adaptive_sequence_loss(student, teacher, base, labels, mask, *, tau=1.0, kd_weight=1.0):
    """CE plus weighted reverse KL, normalized over answer tokens only.

    Gain is log p_16(y) - log p_8_base(y); unlike the original yes/no
    algorithm this variant does not implement foreground, TCD or pair gates.
    """
    if not math.isfinite(tau) or tau <= 0 or not math.isfinite(kd_weight) or kd_weight < 0:
        raise ValueError('tau must be positive and kd_weight nonnegative')
    if student.ndim != 3 or student.shape != teacher.shape or student.shape != base.shape:
        raise ValueError('logits must have matching [batch, tokens, vocab] shapes')
    if labels.shape != student.shape[:2] or mask.shape != labels.shape:
        raise ValueError('labels and mask must match the logits token dimensions')
    valid = mask.bool()
    if not valid.any():
        raise ValueError('answer mask must contain at least one token')
    y = labels[valid].long()
    if (y < 0).any() or (y >= student.shape[-1]).any():
        raise ValueError('answer token outside vocabulary')
    s = F.log_softmax(student[valid].float(), dim=-1)
    t = F.log_softmax(teacher.detach()[valid].float(), dim=-1)
    b = F.log_softmax(base.detach()[valid].float(), dim=-1)
    gain = (t.gather(-1, y[:, None]) - b.gather(-1, y[:, None])).squeeze(-1)
    weights = (gain / tau).clamp(0, 1)
    ce = -s.gather(-1, y[:, None]).mean()
    reverse_kl = (s.exp() * (s - t)).sum(-1)
    kd = (weights * reverse_kl).mean()
    return ce + kd_weight * kd, {'ce': ce.detach(), 'kd': kd.detach(),
                                  'mean_weight': weights.mean().detach(),
                                  'answer_tokens': valid.sum().detach()}
