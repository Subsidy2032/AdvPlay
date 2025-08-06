# advplay/command_dispatcher/handlers/prompt_injection_handlers.py

from advplay.attack_templates.template_registry.registry import define_template
from advplay.attacks.attack_runner import attack_runner
from advplay.utils.list_templates import list_template_names, list_template_contents
from advplay.variables import commands, available_attacks

from advplay.command_dispatcher.handler_registry import register_handler

@register_handler(commands.SAVE_TEMPLATE, available_attacks.PROMPT_INJECTION)
def handle_save_template_prompt_injection(args):
    if args.list:
        if args.template:
            list_template_contents(available_attacks.PROMPT_INJECTION, args.template)
        else:
            list_template_names(available_attacks.PROMPT_INJECTION)
        return

    if args.platform:
        kwargs = {
            "model": args.model,
            "instructions": args.instructions,
        }
        if getattr(args, "filename", None):
            kwargs["filename"] = args.filename

        define_template(args.platform, available_attacks.PROMPT_INJECTION, **kwargs)

@register_handler(commands.ATTACK, available_attacks.PROMPT_INJECTION)
def handle_attack_prompt_injection(args):
    if not args.configuration:
        return

    kwargs = {}
    if getattr(args, "filename", None):
        kwargs["filename"] = args.filename

    attack_runner(args.attack_type, args.configuration, **kwargs)
