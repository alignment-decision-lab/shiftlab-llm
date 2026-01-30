import random
from shiftlab.core.registry import DATASETS
from shiftlab.core.types import DomainData

@DATASETS.register("toy_domains")
def make_toy_domains(n: int = 50, seed: int = 0, **kwargs):
    rng = random.Random(seed)

    # source: formal/modern
    src = [f"Please provide the result of {rng.randint(1,9)} + {rng.randint(1,9)}."
           for _ in range(n)]

    # target: informal / slangy / noisy
    tgt = [f"yo what's {rng.randint(1,9)}+{rng.randint(1,9)}??? lol"
           for _ in range(n)]

    return DomainData("source", src), DomainData("target", tgt)
