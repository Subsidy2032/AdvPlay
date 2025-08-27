from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
import os
import joblib

from advplay.model_ops.trainers.sklearn.sklearn_trainer import SKLearnTrainer
from advplay.variables import available_training_algorithms, available_frameworks
from advplay import paths

class LogisticRegressionTrainer(SKLearnTrainer, framework=available_frameworks.SKLEARN,
                                training_algorithm=available_training_algorithms.LOGISTIC_REGRESSION):
    def __init__(self, X_train, y_train, config: dict = None):
        super().__init__(X_train, y_train, config)

    def train(self):
        model = LogisticRegression()
        model.fit(self.X_train, self.y_train)

        return model
