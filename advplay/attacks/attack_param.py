from dataclasses import dataclass
from typing import Any


@dataclass
class Param:
    type: Any = None
    required: bool = False
    default: Any = None
    help: str = ""
    choices: Any = None


@dataclass
class TemplateParam(Param):
    """Parameter stored in a saved attack template file."""


@dataclass
class AttackParam(Param):
    """Parameter supplied when running an attack."""