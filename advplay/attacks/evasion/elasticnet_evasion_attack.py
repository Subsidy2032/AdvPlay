from typing import Annotated

from art.attacks.evasion import ElasticNet

from advplay.attacks.attack_param import AttackParam
from advplay.attacks.evasion.evasion_attack import EvasionAttack
from advplay.variables import available_attacks, evasion_techniques

class ElasticNetEvasionAttack(EvasionAttack, attack_type=available_attacks.EVASION, attack_subtype=evasion_techniques.ELASTICNET):
    confidence: Annotated[float, AttackParam(type=float, required=False, default=8,
                                      help="How further should the examples be from the source class")]
    learning_rate: Annotated[float, AttackParam(type=float, required=False, default=0.01, help="The learning rate for the attack")]
    binary_search_steps: Annotated[int, AttackParam(type=int, required=False, default=9, help="Numbers of binary steps to optimize the perturbation")]
    max_iter: Annotated[int, AttackParam(type=int, required=False, default=100, help="Maximum iterations")]
    beta: Annotated[float, AttackParam(type=float, required=False, default=0.001, help="Higher value for sparser perturbations (L1-L2 tradeoff)")]
    initial_const: Annotated[float, AttackParam(type=float, required=False, default=0.001, help="Initial constant c for perturbation optimization")]
    batch_size: Annotated[int, AttackParam(type=int, required=False, default=1, help="The batch size")]
    decision_rule: Annotated[str, AttackParam(type=str, required=False, default='EN', help="Decision rule")]

    def execute(self):
        super().execute()

        return self.art_evasion(ElasticNet, confidence=self.confidence, learning_rate=self.learning_rate, binary_search_steps=self.binary_search_steps, 
                                max_iter=self.max_iter, beta=self.beta, initial_const=self.initial_const, batch_size=self.batch_size, decision_rule=self.decision_rule)
