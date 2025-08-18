from pathlib import Path

from advplay.attack_templates.template_builders.template_builder_base import TemplateBuilderBase
from advplay.variables import poisoning_techniques, default_template_file_names

class LabelFlippingTemplateBuilder(TemplateBuilderBase, template_type=poisoning_techniques.LABEL_FLIPPING):
    def __init__(self, attack_type: str, **kwargs):
        super().__init__(attack_type, **kwargs)
        self.training_framework = self.kwargs.get("training_framework")
        self.training_algorithm = self.kwargs.get("training_algorithm")
        self.training_config = self.kwargs.get("training_config")
        self.test_portion = self.kwargs.get("test_portion")
        self.min_portion_to_poison = self.kwargs.get("min_portion_to_poison")
        self.max_portion_to_poison = self.kwargs.get("max_portion_to_poison")
        self.source_class = self.kwargs.get("source_class")
        self.target_class = self.kwargs.get("target_class")
        self.trigger_pattern = self.kwargs.get("trigger_pattern")
        self.override = self.kwargs.get("override")
        self.filename = self.kwargs.get("filename", default_template_file_names.LABEL_FLIPPING)

    def build(self):
        if self.training_config and Path(self.training_config).exists():
            with open(self.training_config, 'r', encoding='utf-8') as config_file:
                self.training_config = config_file.read()

        template = {
            "poisoning_method": poisoning_techniques.LABEL_FLIPPING,
            "training_framework": self.training_framework,
            "training_algorithm": self.training_algorithm,
            "training_config": self.training_config,
            "override": self.override
        }

        self.save_template(self.filename, template)
