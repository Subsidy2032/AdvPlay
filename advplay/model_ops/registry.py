from pathlib import Path

from advplay.model_ops.dataset_loaders.base_dataset_loader import BaseDatasetLoader
from advplay.model_ops.trainers.base_trainer import BaseTrainer
from advplay.model_ops.model_loaders.base_model_loader import BaseModelLoader
from advplay.model_ops.evaluators.base_evaluator import BaseEvaluator
from advplay.utils import load_files
from advplay import paths

def load_dataset(source_type, path, label_column):
    if not Path(path).is_file():
        path = paths.DATASETS / f"{path}.csv"
        if not Path(path).is_file():
            raise FileNotFoundError(f"File {path} does not exist")

    loader_cls = BaseDatasetLoader.registry.get(source_type)

    if loader_cls is None:
        raise ValueError(f"Unsupported source type: {source_type}")

    loader = loader_cls(path, label_column)
    dataset = loader.load()
    return dataset

def build_trainer_cls(framework: str, training_algorithm: str, X_train, y_train, config: dict = None):
    default_path = paths.TRAINING_CONFIGURATIONS / framework
    if isinstance(config, str):
        config = load_files.load_json(default_path, config)

    if not isinstance(config, dict):
        raise TypeError(f"Config must be a JSON object (dict), got {type(config).__name__}")

    key = (framework, training_algorithm)
    trainer_cls = BaseTrainer.registry.get(key)
    if trainer_cls is None:
        raise ValueError(f"Unsupported framework + algorithm: {key}")

    trainer = trainer_cls(X_train, y_train, config)
    return trainer

def train(framework: str, training_algorithm: str, X_train, y_train, config: dict = None):
    trainer = build_trainer_cls(framework, training_algorithm, X_train, y_train, config)
    print(f"Training a model using the {training_algorithm} training algorithm")
    return trainer.train()

def load_model(framework: str, model_path: str):
    default_path = paths.MODELS

    if not Path(model_path).is_file():
        model_path = default_path / model_path
        if not model_path.is_file():
            raise FileNotFoundError(f"model path not found: {model_path}")

    loader_cls = BaseModelLoader.registry.get(framework)

    if loader_cls is None:
        raise ValueError(f"Unsupported framework: {framework}")

    loader = loader_cls(model_path)
    print(f"Loading model: {model_path}")
    return loader.load()

def evaluate_model_accuracy(framework: str, model, X, y):
    evaluator_cls = BaseEvaluator.registry.get(framework)

    if evaluator_cls is None:
        raise ValueError(f"Unsupported framework: {framework}")

    evaluator = evaluator_cls(model)
    print(f"Evaluating model accuracy")
    accuracy = evaluator.accuracy(X, y)
    return accuracy
