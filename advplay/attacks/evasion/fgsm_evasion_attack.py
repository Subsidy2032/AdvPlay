from art.attacks.evasion import FastGradientMethod

from advplay.utils.append_log_entry import append_log_entry
from advplay.attacks.evasion.evasion_attack import EvasionAttack
from advplay.variables import available_attacks, evasion_techniques
from advplay.model_ops import registry

class FGSMEvasionAttack(EvasionAttack, attack_type=available_attacks.EVASION, attack_subtype=evasion_techniques.FGSM):
    ATTACK_PARAMETERS = {
        **EvasionAttack.ATTACK_PARAMETERS,
        "eps": {"type": float, "required": False, "default": 0.01,
                "help": "Maximum perturbation allowed"}
    }

    def __init__(self, template, **kwargs):
        super().__init__(template, **kwargs)

    def execute(self):
        super().execute()

        wrapper = registry.load_classifier(self.training_framework, self.model_path, self.model_configuration)
        perturbed_samples = self.art_evasion(FastGradientMethod, wrapper, eps=self.eps)

        dataset_path = self.save_perturbed_dataset(perturbed_samples)

        results = {
            "original_dataset_path": self.samples.metadata["dataset_path"],
            "perturbed_dataset_path": str(dataset_path)
        }

        self.log_attack_results(results, self.log_file_path)

        return perturbed_samples

    def log_attack_results(self, results, log_file_path):
        log_entry = {
            "attack": self.attack_type,
            "technique": self.attack_subtype,
            "original_dataset_path": results["original_dataset_path"],
            "perturbed_dataset_path": results["perturbed_dataset_path"]
        }

        append_log_entry(log_file_path, log_entry)
