from advplay.model_ops.trainers.base_trainer import BaseTrainer

def build_cls(training_algorithm: str, model_name: str, dataset, label_column:str, test_portion: float, seed: int):
    trainer_cls = BaseTrainer.registry.get(training_algorithm)
    if trainer_cls is None:
        raise ValueError(f"Unsupported training algorithm: {training_algorithm}")

    trainer = trainer_cls(model_name, dataset, label_column, test_portion, seed)
    return trainer

def train(training_algorithm: str, model_name: str, dataset, label_column: str, test_portion: float, seed: int = None):
    trainer = build_cls(training_algorithm, model_name, dataset, label_column, test_portion, seed)
    print(f"Training the model {model_name} using the {training_algorithm} training algorithm")
    trainer.train()

def predict(training_algorithm: str, model_name: str, dataset, label_column: str, test_portion: float, seed: int = None):
    trainer = build_cls(training_algorithm, model_name, dataset, label_column, test_portion, seed)
    trainer.predict()
