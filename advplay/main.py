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
                      (commands.COMMAND, commands.ATTACK_TYPE, 'list', 'template')}

        for key, value in parameters.items():
            type = BaseAttack.registry.get((attack_type, None)).TEMPLATE_PARAMETERS[key].get("type")
            parameters[key] = cast_parameter(value, type, parameters)

        attack_subtype = kwargs.get('technique')
        attack_runner.define_template(attack_type, attack_subtype, **parameters)

    elif args.command == commands.ATTACK:
        parameters = {k: v for k, v in kwargs.items() if k not in (commands.COMMAND, commands.ATTACK_TYPE, 'template')}
        for key, value in parameters.items():
            type = BaseAttack.registry.get((attack_type, None)).ATTACK_PARAMETERS[key].get("type")
            parameters[key] = cast_parameter(value, type, parameters)

        template_name = args.template
        attack_runner.attack_runner(attack_type, template_name, **parameters)

    elif args.command == commands.VISUALIZE:
        parameters = {k: v for k, v in kwargs.items() if k not in (commands.COMMAND, commands.ATTACK_TYPE)}
        visualizer(attack_type, **parameters)

def cast_parameter(parameter, type, parameters):
    if parameter is None or type is None:
        return parameter

    if type == np.array:
        if parameter is None:
            return None

        ext = os.path.splitext(parameter)[1][1:]

        if isinstance(parameters['label_column'], str):
            path = parameter
            if not Path(path).is_file():
                path = paths.DATASETS / f"{path}"
                if not Path(path).is_file():
                    raise FileNotFoundError(f"File {path} does not exist")

            if not ext == 'csv':
                raise TypeError("str column name is currently only supported for csv files")
            df = pd.read_csv(path)
            parameters['label_column'] = df.columns.get_loc(parameters['label_column'])

        return load_dataset(ext, parameter)

    elif type == 'label':
        return parameter

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
            return [parameter]

        except:
            return str(parameter)

    elif type == bool:
        return parameter.lower() == "true"

    else:
        return type(parameter)