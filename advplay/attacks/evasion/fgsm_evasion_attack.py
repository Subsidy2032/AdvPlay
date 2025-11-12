from art.attacks.evasion import FastGradientMethod

from advplay.attacks.evasion.evasion_attack import EvasionAttack
from advplay.variables import available_attacks, evasion_techniques

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
        perturbed_samples = self.art_evasion(FastGradientMethod, eps=self.eps)

        self.log_art_attack_results(perturbed_samples)
        return perturbed_samples
