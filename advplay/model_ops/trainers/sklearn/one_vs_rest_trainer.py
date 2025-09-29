from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier

from advplay.model_ops.trainers.sklearn.sklearn_trainer import SKLearnTrainer
from advplay.variables import available_models, available_frameworks

class OneVsRestTrainer(SKLearnTrainer, framework=available_frameworks.SKLEARN,
                       model=available_models.ONE_VS_REST):
    def __init__(self, X_train, y_train, config: dict = None):
        super().__init__(X_train, y_train, config)
        base_model = LogisticRegression(C=1.0, solver='liblinear')
        self.model = OneVsRestClassifier(base_model)
