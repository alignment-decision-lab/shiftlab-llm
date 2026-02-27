from dataclasses import dataclass
from typing import List
from shiftlab.core.registry import ADAPTERS

@dataclass
class TemperatureScaling:
    T: float = 1.0
    name: str = "temperature_scaling"

    def apply(self, texts: List[str]) -> List[str]:
        # placeholder: does nothing until you plug in a model producing logits/probs
        return texts

@ADAPTERS.register("temperature_scaling")
def make_temperature_scaling(T: float = 1.0, **kwargs):
    return TemperatureScaling(T=float(T))