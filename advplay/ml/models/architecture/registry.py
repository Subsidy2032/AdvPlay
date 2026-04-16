MODEL_REGISTRY = {}

def register_model(cls):
    MODEL_REGISTRY[cls.__name__] = cls
    return cls