from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent

HANDLERS = PROJECT_ROOT / 'command_dispatcher'

ATTACK_TEMPLATES = PROJECT_ROOT / 'attack_templates'
TEMPLATE_BUILDERS = ATTACK_TEMPLATES / 'template_builders'
TEMPLATES = ATTACK_TEMPLATES / 'templates'

ATTACKS = PROJECT_ROOT / 'attacks'
ATTACK_LOGS = ATTACKS / 'logs'
