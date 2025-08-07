from typing import Callable, Dict, Tuple, TypeVar

COMMAND_ATTACK_HANDLERS: Dict[Tuple[str, str], Callable[..., object]] = {}

F = TypeVar('F', bound=Callable[..., object])

def register_handler(command: str, attack_type: str) -> Callable[[F], F]:
    def decorator(func: F) -> F:
        key = (command, attack_type)
        if key in COMMAND_ATTACK_HANDLERS:
            raise ValueError(f"Handler for {key} already registered.")
        COMMAND_ATTACK_HANDLERS[key] = func
        return func
    return decorator
