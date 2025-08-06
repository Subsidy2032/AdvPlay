from advplay.attack_templates.template_registry.registry import define_template
from advplay.variables import parser_names
from advplay.utils.list_templates import list_template_names, list_template_contents
from advplay.paths import LLM_TEMPLATES

def perform_action(args):
    if args.command == parser_names.SAVE_TEMPLATE:
        if args.attack_type == parser_names.LLM:
            if args.list:
                if args.template:
                    list_template_contents(parser_names.LLM, args.template)
                else:
                    list_template_names(parser_names.LLM)
                return

            elif args.platform:
                kwargs = {
                    "model": args.model,
                    "instructions": args.instructions,
                }
                if hasattr(args, "filename") and args.filename:
                    kwargs["filename"] = args.filename

                define_template(args.platform, **kwargs)
