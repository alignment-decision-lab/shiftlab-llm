from dataclasses import dataclass
from typing import Dict, List

@dataclass
class DomainData:
    name: str
    texts: List[str]

@dataclass
class ShiftReport:
    metrics: Dict[str, float]
