from abc import ABC, abstractmethod

from advplay.ml.data.dataset_loaders.loaded_dataset import LoadedDataset


class BasePreprocessor(ABC):
    registry = {}

    def __init_subclass__(cls, name: str, **kwargs):
        super().__init_subclass__(**kwargs)
        if name in BasePreprocessor.registry:
            raise ValueError(f"Preprocessor already registered for name '{name}'")
        cls.name = name
        BasePreprocessor.registry[name] = cls

    def __init__(self, **kwargs):
        for key, value in kwargs.items():
            setattr(self, key, value)

    @abstractmethod
    def apply(self, dataset: LoadedDataset) -> LoadedDataset:
        raise NotImplementedError("Subclasses must implement the apply method")
