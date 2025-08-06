from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent

ATTACK_TEMPLATES = PROJECT_ROOT / 'attack_templates'
TEMPLATE_BUILDERS = ATTACK_TEMPLATES / 'template_builders'
TEMPLATES = ATTACK_TEMPLATES / 'templates'
ATTACKS = PROJECT_ROOT / 'attacks'
HANDLERS = PROJECT_ROOT / 'command_dispatcher'