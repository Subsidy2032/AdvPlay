from pathlib import Path

from advplay.attack_templates.template_builders.poisoning.poisoning_template_builder import PoisoningTemplateBuilder
from advplay.model_ops.trainers.base_trainer import BaseTrainer
from advplay.variables import poisoning_techniques, default_template_file_names, available_attacks

class LabelFlippingTemplateBuilder(PoisoningTemplateBuilder, attack_type=available_attacks.POISONING, template_type=poisoning_techniques.LABEL_FLIPPING):
    def __init__(self, attack_type: str, **kwargs):
        super().__init__(attack_type, **kwargs)

    def build(self):
        super().build()