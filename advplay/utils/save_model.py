import os

from advplay.variables import available_frameworks
from advplay import paths
import joblib
import torch

def save_sklearn_model(model, model_name):
    model_path = paths.MODELS / available_frameworks.SKLEARN / f"{model_name}.joblib"
    os.makedirs(model_path.parent, exist_ok=True)

    joblib.dump(model, model_path)

    print(f"Model saved to {model_path}")
    return model_path

def save_pytorch_model(model, model_name):
    model_path = paths.MODELS / available_frameworks.PYTORCH / f"{model_name}.pth"
    os.makedirs(model_path.parent, exist_ok=True)

    torch.save(model, model_path)

    print(f"Model saved to {model_path}")
    return model_path

def save_model(framework, model, model_name):
    if framework == available_frameworks.SKLEARN:
        return save_sklearn_model(model, model_name)

    elif framework == available_frameworks.PYTORCH:
        return save_pytorch_model(model, model_name)

    else:
        raise ValueError(f"Unsupported framework: {framework}")