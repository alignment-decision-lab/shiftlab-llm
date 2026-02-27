class Registry:
    def __init__(self, name: str = "registry"):
        self.name = name
        self._items = {}

    def register(self, name: str):
        """
        Decorator to register a factory function under a given name.
        Usage:

            @REGISTRY.register("my_component")
            def make_my_component(...):
                return ...
        """
        def decorator(fn):
            if name in self._items:
                raise KeyError(f"[{self.name}] Duplicate registration: '{name}'")
            self._items[name] = fn
            return fn
        return decorator

    def create(self, name: str, **kwargs):
        """
        Instantiate a registered component by name.
        """
        if name not in self._items:
            raise KeyError(
                f"[{self.name}] Unknown component: '{name}'. "
                f"Available: {self.list()}"
            )
        return self._items[name](**kwargs)

    def list(self):
        """
        Return sorted list of registered component names.
        """
        return sorted(self._items.keys())

    def __contains__(self, name: str):
        return name in self._items

    def __repr__(self):
        return f"<Registry name={self.name} items={self.list()}>"

# --------------------------------------------------
# Global registries
# --------------------------------------------------

DATASETS = Registry("datasets")
SHIFT_METRICS = Registry("shift_metrics")
ADAPTERS = Registry("adapters")
SHIFT_OPERATORS = Registry("shift_operators")