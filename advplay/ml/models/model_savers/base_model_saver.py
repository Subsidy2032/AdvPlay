class BaseModelSaver:
    registry = {}

    def __init_subclass__(cls, framework: str):
        if framework in BaseModelSaver.registry:
            raise ValueError(f"Subclass already registered for {framework}")

        super().__init_subclass__()
        BaseModelSaver.registry[framework] = cls

    def save(self, model, model_name: str):
        raise NotImplementedError("Subclasses must implement the save method.")
