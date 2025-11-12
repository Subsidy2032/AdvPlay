from art.attacks.evasion import CarliniL2Method

from advplay.attacks.evasion.evasion_attack import EvasionAttack
from advplay.variables import available_attacks, evasion_techniques

class FGSMEvasionAttack(EvasionAttack, attack_type=available_attacks.EVASION, attack_subtype=evasion_techniques.C_W):
    ATTACK_PARAMETERS = {
        **EvasionAttack.ATTACK_PARAMETERS,
    }

    def __init__(self, template, **kwargs):
        super().__init__(template, **kwargs)

    def execute(self):
        super().execute()
        perturbed_samples = self.art_evasion(CarliniL2Method)

        self.log_art_attack_results(perturbed_samples)
        return perturbed_samples
