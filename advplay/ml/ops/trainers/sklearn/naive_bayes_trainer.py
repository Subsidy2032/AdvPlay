from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
import numpy as np

from advplay.ml.ops.trainers.sklearn.sklearn_trainer import SKLearnTrainer
from advplay.variables import available_models, available_frameworks
from advplay import paths

class NaiveBayesTrainer(SKLearnTrainer, framework=available_frameworks.SKLEARN,
                                model=available_models.NAIVE_BAYES):
    def __init__(self, X_train, y_train, config: dict = None):
        X_train = np.asarray(X_train).ravel()

        self.vectorizer = CountVectorizer(**config, token_pattern=r'\b\w+\b|[£$€¥]+|\d+|!!+|\?\?+|\.\.+')
        X_train_vec = self.vectorizer.fit_transform(X_train)

        super().__init__(X_train_vec, y_train, config)

        self.model = MultinomialNB()

    def train(self):
        model = super().train()
        model.vectorizer = self.vectorizer
        return model

