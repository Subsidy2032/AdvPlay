import json

from advplay.attacks.base_attack import BaseAttack
from advplay.paths import TEMPLATES
from advplay.utils import load_files

def attack_runner(attack_type: str, template_name: str, **kwargs):
    attack_cls = BaseAttack.registry.get(attack_type)
    if attack_cls is None:
        raise ValueError(f"Unsupported attack type: {attack_type}")

    default_path = TEMPLATES / attack_type
    if isinstance(template_name, str):
        template = load_files.load_json(default_path, template_name)

    else:
        template = template_name

    if not isinstance(template, dict):
        raise TypeError(f"template must be a JSON object (dict), got {type(template).__name__}")

    print(f"Running attack '{attack_type}' with template '{template_name}'")
    attack = attack_cls(template, **kwargs)
    attack.execute()
