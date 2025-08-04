from advplay.attack_templates.template_registery import define_template

def perform_action(args):
    if args.command == 'save_template':
        if args.attack_type == 'llm':
            kwargs = {
                "model": args.model,
                "instructions": args.instructions,
            }
            if hasattr(args, "filename") and args.filename:
                kwargs["filename"] = args.filename

            define_template(args.platform, **kwargs)