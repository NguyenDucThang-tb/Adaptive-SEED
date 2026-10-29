"""Answer-gain adaptive weighting; gains must be computed using TRAIN labels."""
import math


def adaptive_weight(max_weight: float, gain: float, tau: float = 1.0) -> float:
    if not all(math.isfinite(x) for x in (max_weight, gain, tau)):
        raise ValueError("Inputs must be finite")
    if max_weight < 0 or tau <= 0:
        raise ValueError("max_weight >= 0 and tau > 0 required")
    return max_weight * min(1.0, max(0.0, gain / tau))
