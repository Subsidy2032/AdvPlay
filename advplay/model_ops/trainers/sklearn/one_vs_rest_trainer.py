from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier
from sklearn.metrics import accuracy_score, classification_report
import os
import joblib

from advplay.model_ops.trainers.sklearn.sklearn_trainer import SKLearnTrainer
from advplay.variables import available_training_algorithms, available_frameworks
from advplay import paths

class OneVsRestTrainer(SKLearnTrainer, framework=available_frameworks.SKLEARN,
                                training_algorithm=available_training_algorithms.ONE_VS_REST):
    def __init__(self, X_train, y_train, config: dict = None):
        super().__init__(X_train, y_train, config)
        base_model = LogisticRegression(C=1.0, solver='liblinear')
        self.model = OneVsRestClassifier(base_model)
