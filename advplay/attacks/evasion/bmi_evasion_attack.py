from art.attacks.evasion import BasicIterativeMethod

from advplay.attacks.evasion.evasion_attack import EvasionAttack
from advplay.variables import available_attacks, evasion_techniques

class FGSMEvasionAttack(EvasionAttack, attack_type=available_attacks.EVASION, attack_subtype=evasion_techniques.BMI):
    ATTACK_PARAMETERS = {
        **EvasionAttack.ATTACK_PARAMETERS,
        "eps": {"type": float, "required": False, "default": 0.01, "help": "Maximum perturbation allowed"},
        "eps_step": {"type": float, "required": False, "default": 0.001, "help": "Step size"},
        "max_iter": {"type": int, "required": False, "default": 10, "help": "Maximum iterations"},
        "batch_size": {"type": int, "required": False, "default": 1, "help": "Batch size"}
    }

    def __init__(self, template, **kwargs):
        super().__init__(template, **kwargs)

    def execute(self):
        super().execute()
        perturbed_samples = self.art_evasion(BasicIterativeMethod, eps=self.eps, eps_step=self.eps_step,
                                             max_iter=self.max_iter, batch_size=self.batch_size)

        self.log_art_attack_results(perturbed_samples)
        return perturbed_samples
