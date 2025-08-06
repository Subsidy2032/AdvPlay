from advplay.attack_templates.template_registry.registry import define_template
from advplay.attacks.attack_runner import attack_runner
from advplay.variables import commands, available_attacks
from advplay.utils.list_templates import list_template_names, list_template_contents

def perform_action(args):
    if args.command == commands.SAVE_TEMPLATE:
        if args.attack_type == available_attacks.PROMPT_INJECTION:
            if args.list:
                if args.template:
                    list_template_contents(available_attacks.PROMPT_INJECTION, args.template)
                else:
                    list_template_names(available_attacks.PROMPT_INJECTION)
                return

            elif args.platform:
                kwargs = {
                    "model": args.model,
                    "instructions": args.instructions,
                }
                if hasattr(args, "filename") and args.filename:
                    kwargs["filename"] = args.filename

                define_template(args.platform, available_attacks.PROMPT_INJECTION, **kwargs)

    elif args.command == commands.ATTACK:
        if args.attack_type == available_attacks.PROMPT_INJECTION:
            if args.configuration:
                kwargs = {}

                if hasattr(args, "filename") and args.filename:
                    kwargs["filename"] = args.filename

                attack_runner(args.attack_type, args.configuration, **kwargs)
