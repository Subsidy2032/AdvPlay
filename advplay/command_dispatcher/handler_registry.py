COMMAND_ATTACK_HANDLERS = {}

def register_handler(command: str, attack_type: str):
    def decorator(func):
        key = (command, attack_type)
        if key in COMMAND_ATTACK_HANDLERS:
            raise ValueError(f"Handler for {key} already registered.")
        COMMAND_ATTACK_HANDLERS[key] = func
        return func
    return decorator
