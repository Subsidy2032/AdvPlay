LOSS_FUNCTION_REGISTRY = {}

def register_loss_function(framework):
    def decorator(func):
        LOSS_FUNCTION_REGISTRY[framework] = func
        return func
    return decorator