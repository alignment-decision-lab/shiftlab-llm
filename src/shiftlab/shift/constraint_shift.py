import random
from dataclasses import dataclass, replace
from typing import Any

from shiftlab.core.registry import SHIFT_OPERATORS

@dataclass
class TightenBudget:
    factor: float = 0.8
    name: str = "tighten_budget"

    def apply(self, tgt: Any, rng: random.Random) -> Any:
        if not hasattr(tgt, "budget"):
            raise TypeError("Target has no attribute 'budget'")
        return replace(tgt, budget=float(tgt.budget) * float(self.factor))

@SHIFT_OPERATORS.register("tighten_budget")
def make_tighten_budget(factor: float = 0.8, **kwargs):
    return TightenBudget(factor=float(factor))


@dataclass
class ChangePenaltyWeights:
    mult: float = 2.0
    name: str = "change_penalty"

    def apply(self, tgt: Any, rng: random.Random) -> Any:
        if not hasattr(tgt, "penalty"):
            raise TypeError("Target has no attribute 'penalty'")
        return replace(tgt, penalty=float(tgt.penalty) * float(self.mult))

@SHIFT_OPERATORS.register("change_penalty")
def make_change_penalty(mult: float = 2.0, **kwargs):
    return ChangePenaltyWeights(mult=float(mult))