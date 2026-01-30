class Registry:
    def __init__(self):
        self._items = {}

    def register(self, name: str):
        def deco(fn):
            if name in self._items:
                raise KeyError(f"Duplicate registration: {name}")
            self._items[name] = fn
            return fn
        return deco

    def create(self, name: str, **kwargs):
        if name not in self._items:
            raise KeyError(f"Unknown component: {name}. Available: {list(self._items)}")
        return self._items[name](**kwargs)

DATASETS = Registry()
SHIFT_METRICS = Registry()
ADAPTERS = Registry()
