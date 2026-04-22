from pathlib import Path

from advplay.orchestrators.base_orchestrator import BaseOrchestrator
from advplay.attacks.base_attack import BaseAttack
from advplay.attack_evaluators.base_attack_evaluator import BaseAttackEvaluator
from advplay.loggers.base_logger import BaseLogger
from advplay import paths
from advplay.visualization.base_visualizer import BaseVisualizer
from advplay.utils import load_files
from advplay.ml.data.dataset_savers.base_dataset_saver import BaseDatasetSaver
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
            