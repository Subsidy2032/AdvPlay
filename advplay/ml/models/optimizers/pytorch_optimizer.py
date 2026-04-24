import torch.optim as optim

from advplay.ml.models.optimizers.registry import register_optimizer
from advplay.variables import available_frameworks

@register_optimizer(available_frameworks.PYTORCH)
def get_pytorch_optimizer(optimizer, model, learning_rate, weight_decay):
    optimizer_cls = getattr(optim, optimizer)
    return optimizer_cls(model.parameters(), lr=learning_rate, weight_decay=weight_decay)