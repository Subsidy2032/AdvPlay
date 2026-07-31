from typing import Annotated

from art.attacks.evasion import CarliniL0Method

from advplay.attacks.attack_param import AttackParam
from advplay.attacks.evasion.evasion_attack import EvasionAttack
from advplay.variables import available_attacks, evasion_techniques


class CWL0EvasionAttack(EvasionAttack, attack_type=available_attacks.EVASION,
                        attack_subtype=evasion_techniques.CW_L0):
    """Carlini & Wagner L0 attack: a sparse (few-pixel) attack like JSMA, but with a
    ``confidence`` margin. JSMA stops the instant the target barely wins, so its
    perturbations sit on the decision boundary and transfer poorly to unseen models.
    Raising ``confidence`` pushes the target class to win by a margin, which is what
    makes the adversarial survive transfer -- at the cost of a few more changed pixels."""

    confidence: Annotated[float, AttackParam(type=float, required=False, default=0.0,
                    help="Margin (kappa) by which the target class must win. Higher values (e.g. 20-40) "
                         "produce higher-confidence, more transferable examples that change more pixels.")]
    learning_rate: Annotated[float, AttackParam(type=float, required=False, default=0.01,
                    help="Learning rate for the optimizer")]
    binary_search_steps: Annotated[int, AttackParam(type=int, required=False, default=10,
                    help="Number of binary-search steps over the sparsity/misclassification trade-off constant")]
    max_iter: Annotated[int, AttackParam(type=int, required=False, default=10,
                    help="Maximum optimizer iterations per search step")]
    initial_const: Annotated[float, AttackParam(type=float, required=False, default=0.01,
                    help="Initial trade-off constant c between L0 sparsity and misclassification")]
    max_halving: Annotated[int, AttackParam(type=int, required=False, default=5,
                    help="Maximum number of halving steps in the line search")]
    max_doubling: Annotated[int, AttackParam(type=int, required=False, default=5,
                    help="Maximum number of doubling steps in the line search")]
    batch_size: Annotated[int, AttackParam(type=int, required=False, default=1, help="Batch size")]

    def execute(self):
        super().execute()
        return self.art_evasion(CarliniL0Method, confidence=self.confidence,
                                learning_rate=self.learning_rate,
                                binary_search_steps=self.binary_search_steps,
                                max_iter=self.max_iter, initial_const=self.initial_const,
                                max_halving=self.max_halving, max_doubling=self.max_doubling,
                                batch_size=self.batch_size)
