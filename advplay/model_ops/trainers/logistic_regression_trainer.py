from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

from advplay.model_ops.trainers.base_trainer import BaseTrainer
from advplay.variables import available_training_algorithms

class OpenAITemplateBuilder(BaseTrainer, training_algorithm=available_training_algorithms.LOGISTIC_REGRESSION):
    def __init__(self, model_name: str, dataset, label_column: str, test_portion: float, seed):
        super().__init__(model_name, dataset, label_column, test_portion, seed)

    def train(self):
        X = self.dataset.loc[:, self.dataset.columns != self.label_column]
        y = self.dataset[self.label_column].to_numpy().ravel()
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=self.test_portion, random_state=self.seed)

        model = LogisticRegression(random_state=self.seed)
        model.fit(X_train, y_train)

        return model, X_test, y_test
