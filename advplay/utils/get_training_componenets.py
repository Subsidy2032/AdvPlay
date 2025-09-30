import torch
import torch.nn as nn
import torch.optim as optim
from advplay.variables import available_frameworks

OPTIMIZER_REGISTRY = {}
LOSS_REGISTRY = {}

def register_optimizer(framework):
    def decorator(func):
        OPTIMIZER_REGISTRY[framework] = func
        return func
    return decorator

def register_loss(framework):
    def decorator(func):
        LOSS_REGISTRY[framework] = func
        return func
    return decorator

@register_optimizer(available_frameworks.PYTORCH)
def get_pytorch_optimizer(optimizer, model, learning_rate):
    optimizer_cls = getattr(optim, optimizer)
    return optimizer_cls(model.parameters(), lr=learning_rate)

@register_loss(available_frameworks.PYTORCH)
def get_pytorch_loss_function(loss_function):
    loss_fn_cls = getattr(nn, loss_function)
    return loss_fn_cls()

def get_optimizer(framework, optimizer, model=None, learning_rate=None):
    return OPTIMIZER_REGISTRY[framework](optimizer, model, learning_rate)

def get_loss_function(framework, loss_function):
    return LOSS_REGISTRY[framework](loss_function)
