import sys

from dotenv import load_dotenv
load_dotenv()

import argparse

from advplay.main import perform_action
from advplay.variables import commands, available_attacks
from advplay.utils.load_classes import load_required_classes
from advplay.model_ops.trainers.base_trainer import BaseTrainer
from advplay.attacks.base_attack import BaseAttack
from advplay.visualization.base_visualizer import BaseVisualizer

def main():
    load_required_classes()

    attacks = BaseAttack.registry.keys()
    unique_attack_categories = list({x[0] for x in attacks})

    parser = argparse.ArgumentParser(description="Practice machine learning attacks from the command line")
    subparsers = parser.add_subparsers(dest=commands.COMMAND, help='Type of action to make')

    save_template_parser = subparsers.add_parser(commands.SAVE_TEMPLATE, help='Save attack configuration templates')
    save_template_subparsers = save_template_parser.add_subparsers(dest=commands.ATTACK_TYPE, help='Types of attacks')

    attack_parser = subparsers.add_parser(commands.ATTACK, help='Run attacks from templates')
    attack_subparsers = attack_parser.add_subparsers(dest=commands.ATTACK_TYPE, help='Types of attacks')
    for attack in unique_attack_categories:
        cls = BaseAttack.registry[(attack, None)]

        save_template_subparser = save_template_subparsers.add_parser(attack, help=f"Add template for {attack} attack")
        save_template_subparser.add_argument('--list', action='store_true',
                                             help=f'List available {attack} templates')
        save_template_subparser.add_argument('--template', required=False,
                                             help='List specific template contents')
        for parameter, arguments in cls.TEMPLATE_PARAMETERS.items():
            choices = arguments.get("choices")
            if callable(choices):
                choices = choices()

            save_template_subparser.add_argument(f"--{parameter.replace('_', '-')}",
                                                 required=arguments["required"] and not ('--list' in sys.argv),
                                                 help=arguments["help"], choices=choices)

        attack_subparser = attack_subparsers.add_parser(attack, help=f"{attack} attack")
        technique_subparsers = attack_subparser.add_subparsers(dest=commands.TECHNIQUE, help='Specific attack technique')
        techniques = [key[1] for key in BaseAttack.registry.keys() if key[0] == attack and key[1] is not None]

        for technique in techniques:
            technique_class = BaseAttack.registry[(attack, technique)]
            technique_parser = technique_subparsers.add_parser(technique, help=f"Run a {technique} attack")


            for parameter, arguments in technique_class.ATTACK_PARAMETERS.items():
                choices = arguments.get("choices")
                if callable(choices):
                    choices = choices()

                technique_parser.add_argument(f"--{parameter.replace('_', '-')}",
                                              required=arguments["required"], help=arguments["help"],
                                              choices=choices)

    visualize_parser = subparsers.add_parser(commands.VISUALIZE, help='Visualize attack results')
    visualize_subparsers = visualize_parser.add_subparsers(dest=commands.ATTACK_TYPE, help='Type of attack')

    visualizers = BaseVisualizer.registry.keys()
    unique_visualizers = list({x[0] for x in attacks})

    for visualizer in unique_visualizers:
        visualize_subparser = visualize_subparsers.add_parser(visualizer, help=f"Visualize {visualizer} attack")
        visualize_subparser.add_argument('--log-filename', required=True, help='Attack log file')
        visualize_subparser.add_argument('--directory', required=False,
                                      help='Name of the directory the results will be saved to')

    perform_action(parser.parse_args())

if __name__ == "__main__":
    main()
