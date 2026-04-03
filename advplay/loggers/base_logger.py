from abc import ABC, abstractmethod

class BaseLogger(ABC):
    def __init__(self, location: str):
        self.location = location

    @abstractmethod
    def log(self, results: dict):
        raise NotImplementedError("Subclasses must implement the log method.")
