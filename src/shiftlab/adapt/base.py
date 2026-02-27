from typing import List, Protocol

class CorrectionModule(Protocol):
    name: str
    def apply(self, texts: List[str]) -> List[str]:
        ...