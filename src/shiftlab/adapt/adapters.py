from shiftlab.core.registry import ADAPTERS

@ADAPTERS.register("no_adapt")
def make_no_adapt(**kwargs):
    return NoAdapt()

class NoAdapt:
    name = "no_adapt"
    def apply(self, texts):
        # identity
        return texts

@ADAPTERS.register("normalize_punct")
def make_normalize_punct(**kwargs):
    return NormalizePunct()

class NormalizePunct:
    name = "normalize_punct"
    def apply(self, texts):
        # toy "adaptation": normalize punctuation/noise
        out = []
        for x in texts:
            y = x.replace("???", "?").replace("lol", "").strip()
            out.append(y)
        return out
