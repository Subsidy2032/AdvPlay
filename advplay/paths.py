from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
ADVPLAY = PROJECT_ROOT / 'advplay'

HANDLERS = ADVPLAY / 'command_dispatcher'

ATTACK_TEMPLATES = ADVPLAY / 'attack_templates'
TEMPLATE_BUILDERS = ATTACK_TEMPLATES / 'template_builders'
TEMPLATES = ATTACK_TEMPLATES / 'templates'

ATTACKS = ADVPLAY / 'attacks'
ATTACK_LOGS = ATTACKS / 'logs'

MODEL_OPS = ADVPLAY / 'model_ops'

RESOURCES = ADVPLAY / 'resources'
MODELS = RESOURCES / 'models'
DATASETS = RESOURCES / 'datasets'
TRAINING_CONFIGURATIONS = RESOURCES / 'training_configurations'
