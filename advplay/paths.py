from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

ADVPLAY = PROJECT_ROOT / 'advplay'
ATTACKS = ADVPLAY / 'attacks'
VISUALIZATIONS = ADVPLAY / 'visualization'
MODEL_OPS = ADVPLAY / 'model_ops'

RESOURCES = PROJECT_ROOT / 'resources'
TEMPLATES = RESOURCES / 'templates'
MODELS = RESOURCES / 'models'
DATASETS = RESOURCES / 'datasets'
CONFIGS = RESOURCES / 'configs'

OUTPUTS = PROJECT_ROOT / 'outputs'
LOGS = OUTPUTS / 'logs'
VISUALIZATIONS_RESULTS = OUTPUTS / 'visualization_results'
