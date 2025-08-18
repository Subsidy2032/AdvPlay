from pathlib import Path

from advplay.attack_templates.template_registry.registry import define_template
from advplay.utils.list_templates import list_template_names, list_template_contents
from advplay.variables import commands, available_attacks
from advplay.attacks.attack_runner import attack_runner
from advplay.command_dispatcher.handler_registry import register_handler

@register_handler(commands.SAVE_TEMPLATE, available_attacks.POISONING)
def handle_save_template_poisoning(args):
    if args.list:
        if args.template:
            list_template_contents(available_attacks.POISONING, args.template)
        else:
            list_template_names(available_attacks.POISONING)
        return

    if args.framework:
        kwargs = {
            "framework": args.framework,
            "algorithm": args.algorithm,
            "test_portion": float(args.test_portion),
            "min_portion_to_poison": float(args.min_portion_to_poison),
        }

        # optional ones: only pass if user actually set them
        if args.config:
            kwargs["config"] = args.config
        if args.max_portion_to_poison:
            kwargs["max_portion_to_poison"] = float(args.max_portion_to_poison)
        if args.source:
            kwargs["source"] = int(args.source)
        if args.target:
            kwargs["target"] = int(args.target)
        if args.trigger:
            kwargs["trigger"] = args.trigger
        if args.override:
            kwargs["override"] = args.override
        if args.filename:
            kwargs["filename"] = args.filename

        define_template(args.technique, available_attacks.POISONING, **kwargs)

@register_handler(commands.ATTACK, available_attacks.POISONING)
def handle_attack_poisoning(args):
    if not args.configuration or not args.dataset or not args.label_column:
        return

    kwargs = {
        "dataset": args.dataset,
        "label_column": args.label_column,
    }

    # optional arguments: add only if present
    if getattr(args, "seed", None):
        kwargs["seed"] = args.seed
    if getattr(args, "step", None):
        kwargs["step"] = args.step
    if getattr(args, "model_name", None):
        kwargs["model_name"] = args.model_name
    if getattr(args, "filename", None):
        kwargs["filename"] = args.filename

    attack_runner(args.attack_type, args.configuration, **kwargs)