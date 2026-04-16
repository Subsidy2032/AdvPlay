from typing import Annotated

from art.attacks.evasion import FastGradientMethod

from advplay.attacks.attack_param import AttackParam
from advplay.attacks.evasion.evasion_attack import EvasionAttack
from advplay.variables import available_attacks, evasion_techniques

class FGSMEvasionAttack(EvasionAttack, attack_type=available_attacks.EVASION, attack_subtype=evasion_techniques.FGSM):
    eps: Annotated[float, AttackParam(type=float, required=False, default=0.01,
                                      help="Maximum perturbation allowed")]

    def execute(self):
        super().execute()
        return self.art_evasion(FastGradientMethod, eps=self.eps)
