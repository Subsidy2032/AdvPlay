from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent

ATTACK_TEMPLATES = PROJECT_ROOT / 'attack_templates'
PROMPT_INJECTION_TEMPLATES = ATTACK_TEMPLATES / 'prompt_injection'
