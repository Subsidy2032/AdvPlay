"""Base class for loggers, which persist attack and evaluation results."""
from abc import ABC, abstractmethod

class BaseLogger(ABC):
    """Base class for writing attack results to a destination.

    Args:
        location: Where results are written (e.g. a file path or directory).
    """

    def __init__(self, location: str):
        self.location = location

    @abstractmethod
    def log(self, results: dict):
        """Persist a results entry.

        Args:
            results: The command, attack result, and evaluation data to record.
        """
        raise NotImplementedError("Subclasses must implement the log method.")
