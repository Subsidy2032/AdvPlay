import sys

from dotenv import load_dotenv
load_dotenv()

import argparse

from advplay.attack_templates.template_builders.template_builder_base import TemplateBuilderBase
from advplay.main import perform_action
from advplay.variables import commands, available_attacks
from advplay.utils.load_classes import load_required_classes

def add_save_template_parsers(save_template_parser):
    llm_parser = save_template_parser.add_parser(available_attacks.PROMPT_INJECTION, help='Define a template for LLM attacks')

    group = llm_parser.add_mutually_exclusive_group(required=True)
    group.add_argument('-l', '--list', action='store_true', help='List available LLM templates')
    llm_parser.add_argument('-t', '--template', required=('-t' in sys.argv or '--template' in sys.argv),
                       help='Template name')

    group.add_argument('-p', '--platform', choices=TemplateBuilderBase.registry.keys(),
                            help='The platform of the LLM')
    llm_parser.add_argument('-m', '--model', required=('-p' in sys.argv or '--platform' in sys.argv),
                            help='The name of the model')
    llm_parser.add_argument('-i', '--instructions', required=False, help='Custom instructions for the model')
    llm_parser.add_argument('-f', '--filename', required=False, help='Configuration file name (without extension)')

def add_attack_parsers(attack_parser):
    prompt_injection_parser = attack_parser.add_parser(available_attacks.PROMPT_INJECTION,
                                                       help='Perform prompt injection attacks')

    prompt_injection_parser.add_argument('-c', '--configuration', required=True,
                                         help='Configuration to use for the attack')
    prompt_injection_parser.add_argument('-p', '--prompt', required=False,
                                         help='Provide a prompt or a file with multiple prompts.')
    prompt_injection_parser.add_argument('-s', '--session_id', required=False, help='Define the session ID')
    prompt_injection_parser.add_argument('-f', '--filename', required=False,
                                         help='The file name to save attack results to')

def main():
    load_required_classes()

    parser = argparse.ArgumentParser(description="Practice machine learning attacks from the command line")
    subparsers = parser.add_subparsers(dest=commands.COMMAND, help='Type of action to make')

    save_template_parser = subparsers.add_parser(commands.SAVE_TEMPLATE, help='Save attack configuration templates')
    save_template_subparsers = save_template_parser.add_subparsers(dest=commands.ATTACK_TYPE, help='Types of attacks')
    add_save_template_parsers(save_template_subparsers)

    attack_parser = subparsers.add_parser(commands.ATTACK, help='Run attacks from templates')
    attack_subparsers = attack_parser.add_subparsers(dest=commands.ATTACK_TYPE, help='Types of attacks')
    add_attack_parsers(attack_subparsers)

    perform_action(parser.parse_args())

if __name__ == "__main__":
    main()
