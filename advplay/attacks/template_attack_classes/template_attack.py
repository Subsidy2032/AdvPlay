# TEMPLATE FILE
from abc import ABC
from typing import Annotated

from advplay.attacks.attack_param import TemplateParam
from advplay.attacks.base_attack import BaseAttack
from advplay.variables import default_template_file_names

class TemplateAttack(BaseAttack, ABC, attack_type='template', attack_subtype=None):
    technique: Annotated[str, BaseAttack.COMMON_TEMPLATE_PARAMETERS['technique']('template')]
    # Other template parameters here
    template_filename: Annotated[str, TemplateParam(type=str, required=False,
                                                    default=default_template_file_names.CUSTOM_INSTRUCTIONS,
                                                    help="Template file name")]

    template: Annotated[str, BaseAttack.COMMON_ATTACK_PARAMETERS['template']]
    # Other attack parameters here
    log_filename: Annotated[str, BaseAttack.COMMON_ATTACK_PARAMETERS['log_filename']]

    def __init__(self, template, **kwargs):
        super().__init__(template, **kwargs)

    def execute(self):
        self.validate_attack_inputs()

        # General attack execution logic

    def validate_attack_inputs(self):
        pass

    def validate_template_inputs(self):
        pass

    if __name__ == '__main__':
        pass