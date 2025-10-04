import numpy as np
import pandas as pd
from pathlib import Path
import os
import json

from advplay.variables import commands, available_attacks
from advplay import paths
from advplay.utils import load_files
from advplay.attacks import attack_runner
from advplay.visualization.visualizer import visualizer
from advplay.attacks.base_attack import BaseAttack
from advplay.model_ops.registry import load_dataset
from advplay.utils.list_templates import list_template_names, list_template_contents
from advplay.model_ops.dataset_loaders.loaded_dataset import LoadedDataset

def perform_action(args):
    kwargs = vars(args)
    attack_type = kwargs.get(commands.ATTACK_TYPE)

    if args.command == commands.SAVE_TEMPLATE:
        if args.list:
            if args.template:
                list_template_contents(attack_type, args.template)
            else:
                list_template_names(attack_type)
            return

        parameters = {k: v for k, v in kwargs.items() if k not in
                      (commands.COMMAND, commands.ATTACK_TYPE, commands.TECHNIQUE, 'list', 'template')}

        for key, value in parameters.items():
            type = BaseAttack.registry.get((attack_type, None)).TEMPLATE_PARAMETERS[key].get("type")
            parameters[key] = cast_parameter(value, type)

        attack_runner.define_template(attack_type, **parameters)

    elif args.command == commands.ATTACK:
        parameters = {k: v for k, v in kwargs.items() if k not in (commands.COMMAND, commands.ATTACK_TYPE, commands.TECHNIQUE, 'template')}
        attack_subtype = kwargs.get(commands.TECHNIQUE)

        for key, value in parameters.items():
            type = BaseAttack.registry.get((attack_type, attack_subtype)).ATTACK_PARAMETERS[key].get("type")
            parameters[key] = cast_parameter(value, type)

        template_name = args.template
        attack_runner.attack_runner(attack_type, attack_subtype, template_name, **parameters)

    elif args.command == commands.VISUALIZE:
        parameters = {k: v for k, v in kwargs.items() if k not in (commands.COMMAND, commands.ATTACK_TYPE)}
        visualizer(attack_type, **parameters)

def cast_parameter(parameter, type):
    if parameter is None or type is None:
        return parameter

    if type == LoadedDataset:
        return load_dataset(os.path.splitext(parameter)[1][1:], parameter) if parameter is not None else None

    elif isinstance(type, tuple):
        for t in type:
            try:
                return t(parameter)
            except Exception:
                pass
        raise ValueError(f"Cannot cast {parameter!r} to any of {type}")

    elif type == dict:
        path = Path(parameter)
        if not path.is_file():
            raise FileNotFoundError(f"Config file not found: {parameter}")
        with open(path, "r") as f:
            return json.load(f)

    elif type == list:
        return [parameter]

    elif type == bool:
        return str(parameter).lower() == "true"

    else:
        return type(parameter)
