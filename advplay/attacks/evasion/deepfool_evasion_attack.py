from typing import Annotated

from art.attacks.evasion import DeepFool

from advplay.attacks.attack_param import AttackParam
from advplay.attacks.evasion.evasion_attack import EvasionAttack
from advplay.variables import available_attacks, evasion_techniques

class DeepfoolEvasionAttack(EvasionAttack, attack_type=available_attacks.EVASION, attack_subtype=evasion_techniques.DEEPFOOL):
    eps: Annotated[float, AttackParam(type=float, required=False, default=0.02,
                                      help="Overshoot parameter to ensure the boundry is crossed")]
    nb_grads: Annotated[float, AttackParam(type=int, required=False, default=10, help="The number of to class gradients to compute")]
    max_iter: Annotated[int, AttackParam(type=int, required=False, default=100, help="Maximum iterations")]
    batch_size: Annotated[int, AttackParam(type=int, required=False, default=1, help="Batch size")]

    def execute(self):
        super().execute()

        return self.art_evasion(DeepFool, epsilon=self.eps, nb_grads=self.nb_grads,
                                             max_iter=self.max_iter, batch_size=self.batch_size)
