from advplay.model_ops.trainers.base_trainer import BaseTrainer
from advplay.utils import load_files
from advplay import paths

def build_trainer_cls(framework: str, training_algorithm: str, model_name: str, dataset,
                      label_column:str, test_portion: float, config: dict = None, seed: int = None):
    default_path = paths.TRAINING_CONFIGURATIONS / framework
    if isinstance(config, str):
        config = load_files.load_json(default_path, config)

    if not isinstance(config, dict):
        raise TypeError(f"Config must be a JSON object (dict), got {type(config).__name__}")

    key = (framework, training_algorithm)
    trainer_cls = BaseTrainer.registry.get(key)
    if trainer_cls is None:
        raise ValueError(f"Unsupported framework + algorithm: {key}")

    trainer = trainer_cls(model_name, dataset, label_column, test_portion, config, seed)
    return trainer

def train(framework: str, training_algorithm: str, model_name: str, dataset,
          label_column: str, test_portion: float, config: dict, seed: int = None):
    trainer = build_trainer_cls(framework, training_algorithm, model_name, dataset,
                                label_column, test_portion, config, seed)
    print(f"Training the model {model_name} using the {training_algorithm} training algorithm")
    trainer.train()
