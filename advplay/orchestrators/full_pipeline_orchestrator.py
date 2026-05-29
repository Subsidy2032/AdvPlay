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
from advplay.ml.data.denormalizers.base_denormalizer import BaseDenormalizer
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

        preprocessing_spec = template.get("preprocessing")
        preprocessors = self._build_pipeline(preprocessing_spec, BasePreprocessor, "preprocessor")
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

        denormalization_spec = template.get("denormalization")
        if denormalization_spec is not None:
            denormalizers = self._build_pipeline(denormalization_spec, BaseDenormalizer, "denormalizer")
        else:
            denormalizers = self._build_default_denormalizers(preprocessing_spec)

        if denormalizers:
            datasets = [(self._apply_denormalizers(denormalizers, ds), path) for ds, path in datasets]
            if visualization_context is not None:
                visualization_context.denormalize(denormalizers)

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
    def _extract_entries(spec, label):
        if isinstance(spec, dict):
            return list(spec.items())
        if isinstance(spec, list):
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
                            f"{label} list entries must be a name, a single-key "
                            "{name: params} mapping, or a {'name': ..., 'params': ...} object"
                        )
                else:
                    raise TypeError(
                        f"{label} list entries must be a string or dict, got {type(entry).__name__}"
                    )
            return entries
        raise TypeError(f"{label} spec must be a list or dict, got {type(spec).__name__}")

    @staticmethod
    def _build_pipeline(spec, base_cls, label):
        if not spec:
            return []
        entries = FullPipelineOrchestrator._extract_entries(spec, label.capitalize())
        instances = []
        for name, params in entries:
            cls = base_cls.registry.get(name)
            if cls is None:
                raise ValueError(f"Unknown {label} '{name}'. "
                                 f"Available: {sorted(base_cls.registry.keys())}")
            instances.append(cls(**(params or {})))
        return instances

    @staticmethod
    def _build_default_denormalizers(preprocessing_spec):
        if not preprocessing_spec:
            return []
        entries = FullPipelineOrchestrator._extract_entries(preprocessing_spec, "Preprocessing")
        denormalizers = []
        for name, params in reversed(entries):
            cls = BaseDenormalizer.registry.get(name)
            if cls is None:
                continue
            denormalizers.append(cls(**(params or {})))
        return denormalizers

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

    @staticmethod
    def _apply_denormalizers(denormalizers, dataset):
        for denormalizer in denormalizers:
            dataset = denormalizer.apply(dataset)
        return dataset
