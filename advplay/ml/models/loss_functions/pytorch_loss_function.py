import torch.nn as nn

from advplay.ml.models.loss_functions.registry import register_loss_function
from advplay.variables import available_frameworks

@register_loss_function(available_frameworks.PYTORCH)
def get_pytorch_loss_function(loss_function):
    loss_fn_cls = getattr(nn, loss_function)
    return loss_fn_cls()