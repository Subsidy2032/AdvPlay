# TEMPLATE FILE
from abc import ABC

from advplay.attacks.base_attack import BaseAttack
from advplay.variables import default_template_file_names

class TemplateAttack(BaseAttack, ABC, attack_type='template', attack_subtype=None):
    TEMPLATE_PARAMETERS = {
        "technique": BaseAttack.COMMON_TEMPLATE_PARAMETERS.get('technique')('template'),
        # Other template parameters here
        "template_filename": {"type": str, "required": False,
                              "default": default_template_file_names.CUSTOM_INSTRUCTIONS,
                              "help": "Template file name"}
    }

    ATTACK_PARAMETERS = {
        "template": BaseAttack.COMMON_ATTACK_PARAMETERS.get('template'),
        # Other attack parameters here
        "log_filename": BaseAttack.COMMON_ATTACK_PARAMETERS.get('log_filename')
    }

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