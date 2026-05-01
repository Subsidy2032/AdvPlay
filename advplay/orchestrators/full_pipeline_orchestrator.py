from pathlib import Path

from advplay.orchestrators.base_orchestrator import BaseOrchestrator
from advplay.attacks.base_attack import BaseAttack
from advplay.attack_evaluators.base_attack_evaluator import BaseAttackEvaluator
from advplay.loggers.base_logger import BaseLogger
from advplay import paths
from advplay.visualization.base_visualizer import BaseVisualizer
from advplay.utils import load_files
from advplay.ml.data.dataset_loaders.loaded_dataset import LoadedDataset
from advplay.ml.data.dataset_savers.base_dataset_saver import BaseDatasetSaver
from advplay.ml.data.preprocessors.base_preprocessor import BasePreprocessor
from advplay.ml.models.model_savers.base_model_saver import BaseModelSaver

class FullPipelineOrchestrator(BaseOrchestrator):
    def __init__(self, evaluator: BaseAttackEvaluator, logger: BaseLogger, visualizer: BaseVisualizer):
        self.evaluator = evaluator
        self.logger = logger
        self.visualizer = visualizer

    def run(self, attack_type, attack_subtype, template_name, command, **kwargs):
        default_path = paths.TEMPLATES / attack_type
        if isinstance(template_name, str):
            template = load_files.load_json(default_path, template_name)

        else:
            template = template_name

        preprocessors = self._build_preprocessors(template.get("preprocessing"))
        kwargs = self._apply_preprocessors(preprocessors, kwargs)

        key = (attack_type, attack_subtype)
        attack_cls = BaseAttack.registry.get(key)
        attack = attack_cls(template, **kwargs)
        attack_results, context, datasets = attack.execute()

        evaluation_results = {}
        models = []
        visualization_context = None
        if self.evaluator:
            evaluation_results, models, visualization_context = self.evaluator.evaluate(context)

        log_entry = {
            "command": command,
            "result": attack_results,
            "evaluation": evaluation_results
        }
        self.logger.log(log_entry)

        for loaded_dataset, dataset_path in datasets:
            saver_cls = BaseDatasetSaver.registry.get(loaded_dataset.source_type)
            saver = saver_cls(loaded_dataset.data, loaded_dataset.metadata, Path(dataset_path))
            saver.save()

        for training_framework, model, model_name in models:
            saver = BaseModelSaver.registry.get(training_framework)()
            saver.save(model, model_name)

        if self.visualizer and visualization_context is not None:
            self.visualizer.visualize(visualization_context)

    @staticmethod
    def _build_preprocessors(spec):
        if not spec:
            return []

        if isinstance(spec, dict):
            entries = list(spec.items())
        elif isinstance(spec, list):
            entries = []
            for entry in spec:
                if isinstance(entry, str):
                    entries.append((entry, {}))
                elif isinstance(entry, dict):
                    if "name" in entry:
                        entries.append((entry["name"], entry.get("params", {}) or {}))
                    elif len(entry) == 1:
                        name, params = next(iter(entry.items()))
                        entries.append((name, params or {}))
                    else:
                        raise ValueError(
                            "Preprocessing list entries must be a name, a single-key "
                            "{name: params} mapping, or a {'name': ..., 'params': ...} object"
                        )
                else:
                    raise TypeError(
                        f"Preprocessing list entries must be a string or dict, got {type(entry).__name__}"
                    )
        else:
            raise TypeError(
                f"Preprocessing spec must be a list or dict, got {type(spec).__name__}"
            )

        preprocessors = []
        for name, params in entries:
            cls = BasePreprocessor.registry.get(name)
            if cls is None:
                raise ValueError(f"Unknown preprocessor '{name}'. "
                                 f"Available: {sorted(BasePreprocessor.registry.keys())}")
            preprocessors.append(cls(**(params or {})))
        return preprocessors

    @staticmethod
    def _apply_preprocessors(preprocessors, kwargs):
        if not preprocessors:
            return kwargs
        out = dict(kwargs)
        for key, value in kwargs.items():
            if not isinstance(value, LoadedDataset):
                continue
            for preprocessor in preprocessors:
                value = preprocessor.apply(value)
            out[key] = value
        return out
