import json

from advplay.attacks.base_attack import BaseAttack
from advplay import paths
from advplay.utils import load_files

def define_template(attack_type: str, attack_subtype: str, **kwargs):
    key = (attack_type, attack_subtype)
    attack_cls = BaseAttack.registry.get(key)
    if attack_cls is None:
        raise ValueError(f"Unsupported template type: {attack_subtype}")

    print(f"Creating a template for {attack_type} attack with type {attack_subtype}")

    if kwargs.get("technique") is None:
        kwargs["technique"] = attack_subtype

    builder = attack_cls(kwargs)
    builder.build()

def attack_runner(attack_type: str, template_name, **kwargs):
    default_path = paths.TEMPLATES / attack_type
    if isinstance(template_name, str):
        template = load_files.load_json(default_path, template_name)

    else:
        template = template_name

    if not isinstance(template, dict):
        raise TypeError(f"template must be a JSON object (dict), got {type(template).__name__}")

    attack_subtype = template.get('technique')
    key = (attack_type, attack_subtype)
    attack_cls = BaseAttack.registry.get(key)
    if attack_cls is None:
        raise ValueError(f"Unsupported attack type and platform: {key}")

    print(f"Running attack '{attack_type}' with template '{template_name}'")
    attack = attack_cls(template, **kwargs)
    attack.execute()
