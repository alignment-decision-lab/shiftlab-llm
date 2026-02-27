import random
from shiftlab.core.registry import DATASETS
from shiftlab.core.types import DomainData

@DATASETS.register("toy_domains")
def make_toy_domains(n: int = 50, seed: int = 0, budget: float = 1.0, penalty: float = 1.0, **kwargs):
    rng = random.Random(seed)

    src = [
        f"Please provide the result of {rng.randint(1,9)} + {rng.randint(1,9)}."
        for _ in range(n)
    ]

    tgt = [
        f"yo what's {rng.randint(1,9)}+{rng.randint(1,9)}??? lol"
        for _ in range(n)
    ]

    return (
        DomainData("source", src, budget=float(budget), penalty=float(penalty)),
        DomainData("target", tgt, budget=float(budget), penalty=float(penalty)),
    )