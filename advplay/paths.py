from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent

ATTACK_TEMPLATES = PROJECT_ROOT / 'attack_templates' / 'templates'
OPENAI_TEMPLATES = ATTACK_TEMPLATES / 'openai'
