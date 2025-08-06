import json

from advplay.attacks.base_attack import BaseAttack
from advplay.paths import TEMPLATES

def attack_runner(attack_type, template_name: str, **kwargs):
    attack_cls = BaseAttack.registry.get(attack_type)
    if attack_cls is None:
        raise ValueError(f"Unsupported attack type: {attack_type}")

    template_path = TEMPLATES / attack_type / f"{template_name}.json"

    if not template_path.exists():
        raise FileNotFoundError(f"Template file not found: {template_path}")

    with open(template_path, 'r') as json_file:
        try:
            template_json = json.load(json_file)

        except Exception as e:
            print(f"Error loading template: {e}")
            return

    print(f"Running attack '{attack_type}' with template '{template_name}'")
    attack = attack_cls(template_json, **kwargs)
    attack.execute()
