import sys

from dotenv import load_dotenv
load_dotenv()

import argparse

from advplay.attack_templates.template_builders.template_builder_base import TemplateBuilderBase
from advplay.main import perform_action
from advplay.variables import commands, available_attacks
from advplay.utils.load_classes import load_required_classes
from advplay.model_ops.trainers.base_trainer import BaseTrainer

def add_save_template_pi_parser(save_template_parser):
    prompt_injection_parser = save_template_parser.add_parser(available_attacks.PROMPT_INJECTION, help='Define a template for prompt injection attacks')

    group = prompt_injection_parser.add_mutually_exclusive_group(required=True)
    group.add_argument('-l', '--list', action='store_true', help='List available prompt injection templates')
    prompt_injection_parser.add_argument('-t', '--template', required=False,
                       help='Template name')

    group.add_argument('-p', '--platform', choices=TemplateBuilderBase.registry[available_attacks.PROMPT_INJECTION].keys(),
                            help='The platform of the LLM')
    prompt_injection_parser.add_argument('-m', '--model', required=('-p' in sys.argv or '--platform' in sys.argv),
                            help='The name of the model')
    prompt_injection_parser.add_argument('-i', '--instructions', required=False, help='Custom instructions for the model')
    prompt_injection_parser.add_argument('-f', '--filename', required=False, help='Configuration file name (without extension)')

def add_save_template_poisoning_parser(save_template_parser):
    poisoning_parser = save_template_parser.add_parser(available_attacks.POISONING, help='Define a template for poisoning attacks')

    group = poisoning_parser.add_mutually_exclusive_group(required=True)
    group.add_argument('-l', '--list', action='store_true', help='List available poisoning templates')
    poisoning_parser.add_argument('-t', '--template', required=False,
                                         help='Template name')

    group.add_argument('--technique', choices=TemplateBuilderBase.registry[available_attacks.POISONING].keys(), help='The poisoning attack type')
    poisoning_parser.add_argument('-f', '--framework', choices=[k[0] for k in BaseTrainer.registry.keys()],
                                  help='Framework for training the model')
    poisoning_parser.add_argument('-a', '--algorithm', choices=[k[1] for k in BaseTrainer.registry.keys()],
                                  required=('-f' in sys.argv or '--framework' in sys.argv), help='The training algorithm')
    poisoning_parser.add_argument('-c', '--config', required=False, help='Path to a training configuration file')
    poisoning_parser.add_argument('-tp', '--test-portion', required=('-f' in sys.argv or '--framework' in sys.argv),
                                  help='Portion of the dataset to be used for testing')
    poisoning_parser.add_argument('-min-poison', '--min-portion-to-poison', required=('-f' in sys.argv or '--framework' in sys.argv),
                                  help='Minimum portion from the dataset to poison')
    poisoning_parser.add_argument('-max-poison', '--max-portion-to-poison', required=False,
                                  help='Maximum portion from the dataset to poison')
    poisoning_parser.add_argument('-s', '--source', required=False, help='Source class to poison')
    poisoning_parser.add_argument('--target', required=False, help='Target class')
    poisoning_parser.add_argument('-tr', '--trigger', required=False, help='A trigger to be used for poisoning')
    poisoning_parser.add_argument('-o', '--override', action='store_true', required=False,
                                  help='If this flag is set, the poisoned examples will be overridden')
    poisoning_parser.add_argument('--filename', required=False, help='Configuration file name (without extension)')

def add_attack_pi_parser(attack_parser):
    prompt_injection_parser = attack_parser.add_parser(available_attacks.PROMPT_INJECTION,
                                                       help='Perform prompt injection attacks')

    prompt_injection_parser.add_argument('-c', '--configuration', required=True,
                                         help='Configuration to use for the attack')
    prompt_injection_parser.add_argument('-p', '--prompt', required=False,
                                         help='Provide a prompt or a file with multiple prompts.')
    prompt_injection_parser.add_argument('-s', '--session-id', required=False, help='Define the session ID')
    prompt_injection_parser.add_argument('-f', '--filename', required=False,
                                         help='The file name to save attack results to')

def add_attack_poison_parser(attack_parser):
    poisoning_parser = attack_parser.add_parser(available_attacks.POISONING,
                                                help='Perform poisoning attacks')

    poisoning_parser.add_argument('-c', '--configuration', required=True, help='Configuration to use for the attack')
    poisoning_parser.add_argument('-d', '--dataset', required=True, help='Dataset to poison')
    poisoning_parser.add_argument('--seed', required=False, help='Seed for reproduction')
    poisoning_parser.add_argument('-l', '--label-column', required=True, help='The name of the label column')
    poisoning_parser.add_argument('--step', required=False, help='Incrementing steps to take for poisoning portions')
    poisoning_parser.add_argument('-m', '--model-name', required=False, help='The name of the model for saving')
    poisoning_parser.add_argument('-f', '--filename', required=False, help='The file name to save the attack results to')

def main():
    load_required_classes()

    parser = argparse.ArgumentParser(description="Practice machine learning attacks from the command line")
    subparsers = parser.add_subparsers(dest=commands.COMMAND, help='Type of action to make')

    save_template_parser = subparsers.add_parser(commands.SAVE_TEMPLATE, help='Save attack configuration templates')
    save_template_subparsers = save_template_parser.add_subparsers(dest=commands.ATTACK_TYPE, help='Types of attacks')
    add_save_template_pi_parser(save_template_subparsers)
    add_save_template_poisoning_parser(save_template_subparsers)

    attack_parser = subparsers.add_parser(commands.ATTACK, help='Run attacks from templates')
    attack_subparsers = attack_parser.add_subparsers(dest=commands.ATTACK_TYPE, help='Types of attacks')
    add_attack_pi_parser(attack_subparsers)
    add_attack_poison_parser(attack_subparsers)

    perform_action(parser.parse_args())

if __name__ == "__main__":
    main()
