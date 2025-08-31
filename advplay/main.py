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

def perform_action(args):
    kwargs = vars(args)
    attack_type = kwargs.get(commands.ATTACK_TYPE)

    if args.command == commands.SAVE_TEMPLATE:
        parameters = {k: v for k, v in kwargs.items() if k not in (commands.COMMAND, commands.ATTACK_TYPE)}
        for key, value in parameters.items():
            type = BaseAttack.registry.get((attack_type, None)).TEMPLATE_PARAMETERS[key].get("type")
            parameters[key] = cast_parameter(value, type)

        attack_subtype = kwargs.get('technique')
        attack_runner.define_template(attack_type, attack_subtype, **parameters)

    elif args.command == commands.ATTACK:
        parameters = {k: v for k, v in kwargs.items() if k not in (commands.COMMAND, commands.ATTACK_TYPE, 'template')}
        for key, value in parameters.items():
            type = BaseAttack.registry.get((attack_type, None)).ATTACK_PARAMETERS[key].get("type")
            parameters[key] = cast_parameter(value, type)

        template_name = args.template
        attack_runner.attack_runner(attack_type, template_name, **parameters)

    elif args.command == commands.VISUALIZE:
        parameters = {k: v for k, v in kwargs.items() if k not in (commands.COMMAND, commands.ATTACK_TYPE)}
        visualizer(attack_type, **parameters)

def cast_parameter(parameter, type):
    if parameter is None or type is None:
        return parameter

    if type == pd.DataFrame:
        dataset_path = parameter
        if not Path(dataset_path).is_file():
            dataset_path = paths.DATASETS / parameter
            if not Path(dataset_path).is_file():
                raise ValueError(f"Dataset not found: {dataset_path}")

        ext = os.path.splitext(dataset_path)[1][1:]
        dataset = load_dataset(ext, dataset_path)
        return dataset

    elif type == dict:
        if not parameter.is_file():
            raise FileNotFoundError(f"Config file not found: {parameter}")

        try:
            with open(parameter, "r") as f:
                return json.load(f)

        except Exception as e:
            raise ValueError("File is not a valid json")

    elif type == (str, int):
        try:
            return int(parameter)

        except:
            return str(parameter)


    elif type == (list, str):
        try:
            return list(parameter)

        except:
            return str(parameter)

    elif type == bool:
        return parameter.lower() == "true"

    else:
        return type(parameter)