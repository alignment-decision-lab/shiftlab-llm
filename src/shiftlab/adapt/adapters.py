from dataclasses import dataclass
from typing import List

from shiftlab.core.registry import ADAPTERS
from shiftlab.adapt.base import CorrectionModule


@dataclass
class NoAdapt:
    name: str = "no_adapt"

    def apply(self, texts: List[str]) -> List[str]:
        return texts


@ADAPTERS.register("no_adapt")
def make_no_adapt(**kwargs) -> CorrectionModule:
    return NoAdapt()


@dataclass
class NormalizePunct:
    name: str = "normalize_punct"

    def apply(self, texts: List[str]) -> List[str]:
        out: List[str] = []
        for x in texts:
            y = x.replace("???", "?").replace("lol", "").strip()
            out.append(y)
        return out


@ADAPTERS.register("normalize_punct")
def make_normalize_punct(**kwargs) -> CorrectionModule:
    return NormalizePunct()