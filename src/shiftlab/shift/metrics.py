import math
from collections import Counter
from typing import List, Dict, Any

from shiftlab.core.registry import SHIFT_METRICS


def _tokenize(text: str) -> List[str]:
    return text.lower().replace("?", " ?").replace(".", " .").split()


def _dist(tokens: List[str]) -> Dict[str, float]:
    c = Counter(tokens)
    tot = sum(c.values()) or 1
    return {k: v / tot for k, v in c.items()}


def _jsd(p: Dict[str, float], q: Dict[str, float]) -> float:
    keys = set(p) | set(q)
    m = {k: 0.5 * (p.get(k, 0.0) + q.get(k, 0.0)) for k in keys}

    def _kl(a, b):
        s = 0.0
        for k in keys:
            ak = a.get(k, 0.0)
            bk = b.get(k, 1e-12)
            if ak > 0:
                s += ak * math.log(ak / bk)
        return s

    return 0.5 * _kl(p, m) + 0.5 * _kl(q, m)


# ----------------------------
# Existing metrics (updated signature)
# ----------------------------

@SHIFT_METRICS.register("token_jsd")
def make_token_jsd(**kwargs):
    return TokenJSD()


class TokenJSD:
    name = "token_jsd"

    def compute(self, src: Any, tgt: Any) -> float:
        source_texts = src.texts
        target_texts = tgt.texts
        ps = _dist([t for x in source_texts for t in _tokenize(x)])
        pt = _dist([t for x in target_texts for t in _tokenize(x)])
        return float(_jsd(ps, pt))


@SHIFT_METRICS.register("oov_rate")
def make_oov_rate(**kwargs):
    return OOVRate()


class OOVRate:
    name = "oov_rate"

    def compute(self, src: Any, tgt: Any) -> float:
        source_texts = src.texts
        target_texts = tgt.texts
        src_vocab = set([t for x in source_texts for t in _tokenize(x)])
        tgt_tokens = [t for x in target_texts for t in _tokenize(x)]
        if not tgt_tokens:
            return 0.0
        oov = sum(1 for t in tgt_tokens if t not in src_vocab)
        return float(oov / len(tgt_tokens))


# ----------------------------
# New metric: constraint violation (toy)
# ----------------------------

@SHIFT_METRICS.register("constraint_violation_rate")
def make_constraint_violation_rate(c: float = 100.0, **kwargs):
    return ConstraintViolationRate(c=float(c))


class ConstraintViolationRate:
    """
    Toy constraint: mean length(tgt.texts) <= tgt.budget * c
    Returns 1.0 if violated else 0.0
    """
    name = "constraint_violation_rate"

    def __init__(self, c: float = 100.0):
        self.c = float(c)

    def compute(self, src: Any, tgt: Any) -> float:
        texts = tgt.texts
        if not texts:
            return 0.0
        avg_len = sum(len(x) for x in texts) / len(texts)
        budget = float(getattr(tgt, "budget", 1.0))
        thr = budget * self.c
        return 1.0 if avg_len > thr else 0.0


# ----------------------------
# New metric: "calibration" proxy (cheap, label-free)
# ----------------------------

@SHIFT_METRICS.register("entropy_mean")
def make_entropy_mean(**kwargs):
    return EntropyMean()


class EntropyMean:
    """
    Proxy: mean character-entropy of tgt texts.
    Not true model calibration; cheap drift/uncertainty proxy.
    """
    name = "entropy_mean"

    def compute(self, src: Any, tgt: Any) -> float:
        texts = tgt.texts
        if not texts:
            return 0.0

        def char_entropy(s: str) -> float:
            if not s:
                return 0.0
            c = Counter(s)
            n = sum(c.values())
            ent = 0.0
            for v in c.values():
                p = v / n
                ent -= p * math.log(p + 1e-12)
            return ent

        return float(sum(char_entropy(x) for x in texts) / len(texts))