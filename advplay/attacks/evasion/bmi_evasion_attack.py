from typing import Annotated

from art.attacks.evasion import BasicIterativeMethod

from advplay.attacks.attack_param import AttackParam
from advplay.attacks.evasion.evasion_attack import EvasionAttack
from advplay.variables import available_attacks, evasion_techniques

class BMIEvasionAttack(EvasionAttack, attack_type=available_attacks.EVASION, attack_subtype=evasion_techniques.BMI):
    eps: Annotated[float, AttackParam(type=float, required=False, default=0.01,
                                      help="Maximum perturbation allowed")]
    eps_step: Annotated[float, AttackParam(type=float, required=False, default=0.001, help="Step size")]
    max_iter: Annotated[int, AttackParam(type=int, required=False, default=10, help="Maximum iterations")]
    batch_size: Annotated[int, AttackParam(type=int, required=False, default=1, help="Batch size")]

    def execute(self):
        super().execute()
        return self.art_evasion(BasicIterativeMethod, eps=self.eps, eps_step=self.eps_step,
                                             max_iter=self.max_iter, batch_size=self.batch_size)
