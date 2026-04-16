import os
import joblib

from advplay.ml.models.model_savers.base_model_saver import BaseModelSaver
from advplay.variables import available_frameworks
from advplay import paths

class SklearnModelSaver(BaseModelSaver, framework=available_frameworks.SKLEARN):
    def save(self, model, model_name):
        model_path = paths.MODELS / available_frameworks.SKLEARN / f"{model_name}.joblib"
        os.makedirs(model_path.parent, exist_ok=True)

        joblib.dump(model, model_path)
        return model_path
