import pandas as pd
import numpy as np
import os
from abc import ABC, abstractmethod


class BaseDatasetLoader(ABC):
    registry = {}

    def __init_subclass__(cls, source_type: str):
        key = source_type
        if key in BaseDatasetLoader.registry:
            raise ValueError(f"Loader already registered for source '{source_type}'")
        super().__init_subclass__()
        BaseDatasetLoader.registry[key] = cls

    def __init__(self, path):
        self.path = path

    @abstractmethod
    def load(self) -> pd.DataFrame:
        raise NotImplementedError