from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent

ATTACK_TEMPLATES = PROJECT_ROOT / 'attack_templates'
TEMPLATE_BUILDERS = ATTACK_TEMPLATES / 'template_builders'
TEMPLATES = ATTACK_TEMPLATES / 'templates'
LLM_TEMPLATES = TEMPLATES / 'llm'
