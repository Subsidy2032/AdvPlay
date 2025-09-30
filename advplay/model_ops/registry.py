from pathlib import Path

from advplay.model_ops.dataset_loaders.base_dataset_loader import BaseDatasetLoader
from advplay.model_ops.trainers.base_trainer import BaseTrainer
from advplay.model_ops.model_loaders.base_model_loader import BaseModelLoader
from advplay.model_ops.evaluators.base_evaluator import BaseEvaluator
from advplay.model_ops.dataset_savers.base_dataset_saver import BaseDatasetSaver
from advplay.model_ops.dataset_loaders.loaded_dataset import LoadedDataset
from advplay.utils import load_files, get_training_componenets
from advplay import paths

def load_dataset(source_type, path):
    if not Path(path).is_file():
        path = paths.DATASETS / f"{path}"
        if not Path(path).is_file():
            raise FileNotFoundError(f"File {path} does not exist")

    loader_cls = BaseDatasetLoader.registry.get(source_type)

    if loader_cls is None:
        raise ValueError(f"Unsupported source type: {source_type}")

    loader = loader_cls(path)
    dataset = loader.load()
    return dataset

def save_dataset(loaded_dataset: LoadedDataset, path):
    saver_cls = BaseDatasetSaver.registry.get(loaded_dataset.source_type)

    if saver_cls is None:
        raise ValueError(f"Unsupported source type: {loaded_dataset.source_type}")

    saver = saver_cls(loaded_dataset.data, loaded_dataset.metadata, Path(path))
    saver.save()

def build_trainer_cls(framework: str, model: str, X_train, y_train, config: dict = None):
    default_path = paths.TRAINING_CONFIGURATIONS / framework
    if isinstance(config, str):
        config = load_files.load_json(default_path, config)

    if config is not None and not isinstance(config, dict):
        raise TypeError(f"Config must be a JSON object (dict), got {type(config).__name__}")

    key = (framework, model)
    trainer_cls = BaseTrainer.registry.get(key)
    if trainer_cls is None:
        raise ValueError(f"Unsupported framework + algorithm: {key}")

    trainer = trainer_cls(X_train, y_train, config)
    return trainer

def train(framework: str, model: str, X_train, y_train, config: dict = None):
    trainer = build_trainer_cls(framework, model, X_train, y_train, config)
    print(f"Training a {model} model")
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

def load_classifier(framework, model_path, config: dict):
    default_path = paths.MODELS

    if not Path(model_path).is_file():
        model_path = default_path / model_path
        if not model_path.is_file():
            raise FileNotFoundError(f"model path not found: {model_path}")

    loader_cls = BaseModelLoader.registry.get(framework)

    if loader_cls is None:
        raise ValueError(f"Unsupported framework: {framework}")

    loader = loader_cls(model_path)
    print(f"Loading classifier: {model_path}")

    loss = config.get("loss")
    input_shape = config.get("input_shape")
    nb_classes = config.get("nb_classes")

    loss_fn = get_training_componenets.get_loss_function(framework, loss)
    return loader.load_art_classifier(loss_fn, input_shape, nb_classes)

def evaluate_model_accuracy(framework: str, model, X, y):
    evaluator_cls = BaseEvaluator.registry.get(framework)

    if evaluator_cls is None:
        raise ValueError(f"Unsupported framework: {framework}")

    evaluator = evaluator_cls(model)
    print(f"Evaluating model accuracy")
    accuracy = evaluator.accuracy(X, y)
    return accuracy

def predict(framework: str, model, X):
    evaluator_cls = BaseEvaluator.registry.get(framework)

    if evaluator_cls is None:
        raise ValueError(f"Unsupported framework: {framework}")

    evaluator = evaluator_cls(model)

    print(f"Getting model predictions")
    return evaluator.predict(X)
