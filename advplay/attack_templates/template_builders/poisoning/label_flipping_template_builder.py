from pathlib import Path

from advplay.attack_templates.template_builders.template_builder_base import TemplateBuilderBase
from advplay.model_ops.trainers.base_trainer import BaseTrainer
from advplay.variables import poisoning_techniques, default_template_file_names, available_attacks

class LabelFlippingTemplateBuilder(TemplateBuilderBase, attack_type=available_attacks.POISONING, template_type=poisoning_techniques.LABEL_FLIPPING):
    def __init__(self, attack_type: str, **kwargs):
        super().__init__(attack_type, **kwargs)
        self.training_framework = self.kwargs.get("framework")
        self.training_algorithm = self.kwargs.get("algorithm")
        self.training_config = self.kwargs.get("config")
        self.test_portion = self.kwargs.get("test_portion")
        self.min_portion_to_poison = self.kwargs.get("min_portion_to_poison")
        self.max_portion_to_poison = self.kwargs.get("max_portion_to_poison")
        self.source_class = self.kwargs.get("source")
        self.target_class = self.kwargs.get("target")
        self.trigger_pattern = self.kwargs.get("trigger")
        self.override = self.kwargs.get("override")
        self.filename = self.kwargs.get("filename", default_template_file_names.LABEL_FLIPPING)

        self.validate_parameters()

    def build(self):
        if self.training_config and Path(self.training_config).exists():
            with open(self.training_config, 'r', encoding='utf-8') as config_file:
                self.training_config = config_file.read()

        template = {
            "poisoning_method": poisoning_techniques.LABEL_FLIPPING,
            "training_framework": self.training_framework,
            "training_algorithm": self.training_algorithm,
            "training_config": self.training_config,
            "source_class": self.source_class,
            "target_class": self.target_class,
            "min_portion_to_poison": self.min_portion_to_poison,
            "max_portion_to_poison": self.max_portion_to_poison,
            "test_portion": self.test_portion,
            "trigger_pattern": self.trigger_pattern,
            "override": self.override,
        }

        self.save_template(self.filename, template)

    def validate_parameters(self):
        if (self.training_framework, self.training_algorithm) not in BaseTrainer.registry.keys():
            raise ValueError(f"Invalid framework and training algorithm configuration: ({self.training_framework}, {self.training_algorithm})")

        for name, val in [
            ("test_portion", self.test_portion),
            ("min_portion_to_poison", self.min_portion_to_poison),
            ("max_portion_to_poison", self.max_portion_to_poison)
        ]:
            if val is None:
                continue
            if not isinstance(val, (int, float)):
                raise TypeError(f"{name} must be a number, got {type(val).__name__}")
            if not (0 <= val <= 1):
                raise ValueError(f"{name} must be between 0 and 1, got {val}")

        if (self.min_portion_to_poison is not None
                and self.max_portion_to_poison is not None
                and self.min_portion_to_poison > self.max_portion_to_poison):
            raise ValueError("min_portion_to_poison cannot be greater than max_portion_to_poison")

        if self.source_class is not None and self.target_class is not None:
            if self.source_class == self.target_class:
                raise ValueError("source_class and target_class must be different")

        if self.override is not None and not isinstance(self.override, bool):
            raise TypeError(f"override must be a boolean, got {type(self.override).__name__}")

        if not isinstance(self.filename, str) or not self.filename.strip():
            raise ValueError("filename must be a non-empty string")
        if any(c in self.filename for c in r'\/:*?"<>|'):
            raise ValueError(f"filename contains invalid characters: {self.filename}")

        if self.training_config is not None and not Path(self.training_config).exists():
            raise FileNotFoundError(f"training_config file does not exist: {self.training_config}")