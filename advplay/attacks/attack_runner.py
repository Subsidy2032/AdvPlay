import json

from advplay.attacks.base_attack import BaseAttack
from advplay import paths
from advplay.utils import load_files

def attack_runner(attack_type: str, attack_subtype, template_name, **kwargs):
    key = (attack_type, attack_subtype)
    attack_cls = BaseAttack.registry.get(key)
    if attack_cls is None:
        raise ValueError(f"Unsupported attack type and platform: {key}")

    default_path = paths.TEMPLATES / attack_type
    if isinstance(template_name, str):
        template = load_files.load_json(default_path, template_name)

    else:
        template = template_name

    if not isinstance(template, dict):
        raise TypeError(f"template must be a JSON object (dict), got {type(template).__name__}")

    print(f"Running attack '{attack_type}' with template '{template_name}'")
    attack = attack_cls(template, **kwargs)
    attack.execute()
