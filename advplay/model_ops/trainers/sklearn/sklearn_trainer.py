from sklearn.linear_model import LogisticRegression

from advplay.model_ops.trainers.base_trainer import BaseTrainer
from advplay.variables import available_training_algorithms, available_frameworks
from advplay import paths

class SKLearnTrainer(BaseTrainer, framework=available_frameworks.SKLEARN, training_algorithm=None):
    def __init__(self, X_train, y_train, config: dict = None):
        super().__init__(X_train, y_train, config)
        self.model = None

    def train(self):
        if self.model is None:
            raise NotImplementedError("Subclasses must define self.model before calling train.")

        self.model.fit(self.X_train, self.y_train)
        return self.model
