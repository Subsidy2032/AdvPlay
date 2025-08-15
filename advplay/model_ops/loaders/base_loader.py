class BaseLoader:
    registry = {}

    def __init_subclass__(cls, framework: str):
        if framework in BaseLoader.registry:
            raise ValueError(f"Subclass already registered for {framework}")

        super().__init_subclass__()
        BaseLoader.registry[framework] = cls

    def __init__(self, model_path: str):
        self.model_path = model_path

    def load(self):
        raise NotImplementedError("Subclasses must implement the load method.")
