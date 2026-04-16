OPTIMIZER_REGISTRY = {}

def register_optimizer(framework):
    def decorator(func):
        OPTIMIZER_REGISTRY[framework] = func
        return func
    return decorator
