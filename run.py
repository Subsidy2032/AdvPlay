import sys

from dotenv import load_dotenv
load_dotenv()

import argparse

from advplay.attack_templates.template_builders.template_builder_base import TemplateBuilderBase
from advplay.main import perform_action
from advplay.variables import commands, available_attacks
from advplay.utils.load_classes import load_required_classes
from advplay.model_ops.trainers.base_trainer import BaseTrainer
from advplay.attacks.base_attack import BaseAttack

def add_visualize_poisoning_parser(visualize_parser):
    poisoning_parser = visualize_parser.add_parser(available_attacks.POISONING, help='Visualize poisoning attack results')

    poisoning_parser.add_argument('-f', '--file', required=True, help='Attack log file')
    poisoning_parser.add_argument('-d', '--directory', required=False, help='Name of the directory the results will be saved to')

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
        for parameter, arguments in cls.TEMPLATE_PARAMETERS.items():
            save_template_subparser.add_argument(f"--{parameter.replace('_', '-')}",
                                                 required=arguments["required"], help=arguments["help"])

        attack_subparser = attack_subparsers.add_parser(attack, help=f"{attack} attack")
        for parameter, arguments in cls.ATTACK_PARAMETERS.items():
            attack_subparser.add_argument(f"--{parameter.replace('_', '-')}",
                                          required=arguments["required"], help=arguments["help"])

    visualize_parser = subparsers.add_parser(commands.VISUALIZE, help='Visualize attack results')
    visualize_subparsers = visualize_parser.add_subparsers(dest=commands.ATTACK_TYPE, help='Type of attack')
    add_visualize_poisoning_parser(visualize_subparsers)

    perform_action(parser.parse_args())

if __name__ == "__main__":
    main()
