from advplay.orchestrators.base_orchestrator import BaseOrchestrator
from advplay.attacks.attack_runner import attack_runner
from advplay.attack_evaluators.base_attack_evaluator import BaseAttackEvaluator
from advplay.loggers.base_logger import BaseLogger
from advplay.model_ops import registry
from advplay.utils import save_model

class FullPipelineOrchestrator(BaseOrchestrator):
    def __init__(self, evaluator: BaseAttackEvaluator, logger: BaseLogger):
        self.evaluator = evaluator
        self.logger = logger

    def run(self, attack_type, attack_subtype, template_name, **kwargs):
        attack_results, context, datasets = attack_runner(attack_type, attack_subtype, template_name, **kwargs)

        evaluation_results = {}
        models = []
        if self.evaluator:
            evaluation_results, models = self.evaluator.evaluate(context)

        self.logger.log({
            "result": attack_results,
            "evaluation": evaluation_results
        })

        for loaded_dataset, dataset_path in datasets:
            registry.save_dataset(loaded_dataset, dataset_path)

        for training_framework, model, model_name in models:
            save_model.save_model(training_framework, model, model_name)
