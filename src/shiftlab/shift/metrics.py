import math
from collections import Counter
from typing import List, Dict
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

@SHIFT_METRICS.register("token_jsd")
def make_token_jsd(**kwargs):
    return TokenJSD()

class TokenJSD:
    name = "token_jsd"
    def compute(self, source_texts: List[str], target_texts: List[str]) -> float:
        ps = _dist([t for x in source_texts for t in _tokenize(x)])
        pt = _dist([t for x in target_texts for t in _tokenize(x)])
        return float(_jsd(ps, pt))

@SHIFT_METRICS.register("oov_rate")
def make_oov_rate(**kwargs):
    return OOVRate()

class OOVRate:
    name = "oov_rate"
    def compute(self, source_texts: List[str], target_texts: List[str]) -> float:
        src_vocab = set([t for x in source_texts for t in _tokenize(x)])
        tgt_tokens = [t for x in target_texts for t in _tokenize(x)]
        if not tgt_tokens:
            return 0.0
        oov = sum(1 for t in tgt_tokens if t not in src_vocab)
        return float(oov / len(tgt_tokens))
