from art.attacks.evasion import SaliencyMapMethod

from advplay.attacks.evasion.evasion_attack import EvasionAttack
from advplay.variables import available_attacks, evasion_techniques

class FGSMEvasionAttack(EvasionAttack, attack_type=available_attacks.EVASION, attack_subtype=evasion_techniques.JSMA):
    ATTACK_PARAMETERS = {
        **EvasionAttack.ATTACK_PARAMETERS,
        "theta": {"type": float, "required": False, "default": 0.1,
                  "help": "The amount of perturbation to introduce in each step"},
        "gamma": {"type": float, "required": False, "default": 0.1, "help": "Maximum fraction to effect"},
        "batch_size": {"type": int, "required": False, "default": 1, "help": "Batch size"}
    }

    def __init__(self, template, **kwargs):
        super().__init__(template, **kwargs)

    def execute(self):
        super().execute()
        perturbed_samples = self.art_evasion(SaliencyMapMethod, theta=self.theta, gamma=self.gamma,
                                             batch_size=self.batch_size)

        self.log_art_attack_results(perturbed_samples)
        return perturbed_samples
