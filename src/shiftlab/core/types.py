from dataclasses import dataclass
from typing import Any, Dict, List, Protocol
import random


# ----------------------------
# Dataset abstraction
# ----------------------------

class Domain(Protocol):
    ...


# ----------------------------
# Shift operator
# ----------------------------

class ShiftOperator(Protocol):
    def apply(self, tgt: Any, rng: random.Random) -> Any:
        ...


# ----------------------------
# Adapter / Correction module
# ----------------------------

class Adapter(Protocol):
    def apply(self, texts: List[str]) -> List[str]:
        ...


# ----------------------------
# Metric
# ----------------------------

class Metric(Protocol):
    name: str
    def compute(self, src: Any, tgt: Any) -> float:
        ...
# ----------------------------
# Concrete data container used by toy_domains
# ----------------------------

@dataclass(frozen=True)
class DomainData:
    name: str
    texts: List[str]
    budget: float = 1.0
    penalty: float = 1.0


# ----------------------------
# Report object used by runner
# ----------------------------

@dataclass
class ShiftReport:
    metrics: Dict[str, float]