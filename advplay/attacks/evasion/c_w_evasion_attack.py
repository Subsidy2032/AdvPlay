from art.attacks.evasion import CarliniL2Method

from advplay.attacks.evasion.evasion_attack import EvasionAttack
from advplay.variables import available_attacks, evasion_techniques

class CWEvasionAttack(EvasionAttack, attack_type=available_attacks.EVASION, attack_subtype=evasion_techniques.C_W):
    def execute(self):
        super().execute()
        return self.art_evasion(CarliniL2Method)
