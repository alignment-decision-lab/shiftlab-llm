import random
from dataclasses import dataclass, replace
from typing import Any

from shiftlab.core.registry import SHIFT_OPERATORS


def _inject_punct(s: str, rng: random.Random, p: float) -> str:
    if rng.random() > p:
        return s
    return s.replace(" ", " !!! ")


@dataclass
class InjectPunctShift:
    p: float = 0.3
    name: str = "inject_punct"

    def apply(self, tgt: Any, rng: random.Random) -> Any:
        # If it's your DomainData (frozen dataclass), use replace
        if hasattr(tgt, "texts"):
            new_texts = [_inject_punct(x, rng, self.p) for x in tgt.texts]
            return replace(tgt, texts=new_texts)

        # If it's a dict
        if isinstance(tgt, dict) and "texts" in tgt:
            new = dict(tgt)
            new["texts"] = [_inject_punct(x, rng, self.p) for x in tgt["texts"]]
            return new

        raise TypeError("Unsupported target domain format for InjectPunctShift")


@SHIFT_OPERATORS.register("inject_punct")
def make_inject_punct(p: float = 0.3, **kwargs):
    return InjectPunctShift(p=float(p))