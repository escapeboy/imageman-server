from collections import OrderedDict
import torch


class ModelRegistry:
    """LRU cache for GPU models. Evicts oldest when capacity exceeded."""

    def __init__(self, max_models: int = 3):
        self._cache: OrderedDict = OrderedDict()
        self._max = max_models

    def get(self, name: str):
        if name in self._cache:
            self._cache.move_to_end(name)
            return self._cache[name]
        return None

    def put(self, name: str, model) -> None:
        if len(self._cache) >= self._max:
            oldest = next(iter(self._cache))
            del self._cache[oldest]
            torch.cuda.empty_cache()
        self._cache[name] = model
        self._cache.move_to_end(name)

    @property
    def loaded(self) -> dict:
        return dict(self._cache)


registry = ModelRegistry()
