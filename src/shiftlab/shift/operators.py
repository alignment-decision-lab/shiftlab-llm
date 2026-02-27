import random
from dataclasses import dataclass
from typing import Any

from shiftlab.core.registry import SHIFT_OPERATORS


@dataclass
class NoShift:
    def apply(self, tgt: Any, rng: random.Random) -> Any:
        return tgt


@SHIFT_OPERATORS.register("no_shift")
def make_no_shift(**kwargs):
    return NoShift()