from advplay.command_dispatcher.handler_registry import COMMAND_ATTACK_HANDLERS

def perform_action(args):
    handler = COMMAND_ATTACK_HANDLERS.get((args.command, args.attack_type))
    if handler:
        handler(args)
    else:
        raise NotImplementedError(
            f"No handler for command '{args.command}' and attack type '{args.attack_type}'"
        )