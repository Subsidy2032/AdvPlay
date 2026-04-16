import os
import torch

from advplay.ml.models.model_savers.base_model_saver import BaseModelSaver
from advplay.variables import available_frameworks
from advplay import paths

class PytorchModelSaver(BaseModelSaver, framework=available_frameworks.PYTORCH):
    def save(self, model, model_name):
        model_path = paths.MODELS / available_frameworks.PYTORCH / f"{model_name}.pth"
        os.makedirs(model_path.parent, exist_ok=True)

        torch.save(model.state_dict(), model_path)
        return model_path
