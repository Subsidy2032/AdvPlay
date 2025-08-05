from advplay.attack_templates.template_registery import define_template
from advplay.variables import *

def perform_action(args):
    if args.command == parser_names.SAVE_TEMPLATE:
        if args.attack_type == parser_names.LLM:
            kwargs = {
                "model": args.model,
                "instructions": args.instructions,
            }
            if hasattr(args, "filename") and args.filename:
                kwargs["filename"] = args.filename

            define_template(args.platform, **kwargs)