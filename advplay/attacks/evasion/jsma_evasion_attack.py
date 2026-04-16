from typing import Annotated

from art.attacks.evasion import SaliencyMapMethod

from advplay.attacks.attack_param import AttackParam
from advplay.attacks.evasion.evasion_attack import EvasionAttack
from advplay.variables import available_attacks, evasion_techniques

class JSMAEvasionAttack(EvasionAttack, attack_type=available_attacks.EVASION, attack_subtype=evasion_techniques.JSMA):
    theta: Annotated[float, AttackParam(type=float, required=False, default=0.1,
                                        help="The amount of perturbation to introduce in each step")]
    gamma: Annotated[float, AttackParam(type=float, required=False, default=0.1,
                                        help="Maximum fraction to effect")]
    batch_size: Annotated[int, AttackParam(type=int, required=False, default=1, help="Batch size")]

    def execute(self):
        super().execute()
        return self.art_evasion(SaliencyMapMethod, theta=self.theta, gamma=self.gamma,
                                             batch_size=self.batch_size)
