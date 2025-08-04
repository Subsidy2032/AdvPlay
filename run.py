from dotenv import load_dotenv
load_dotenv()

from advplay.attack_templates.template_registery import TEMPLATE_BUILDERS
from advplay.attack_templates import template_builders
from advplay.main import perform_action

import argparse

def add_parsers(parser):
    llm_parser = parser.add_parser("llm", help='Perform attacks on LLM')
    llm_parser.add_argument('-p', '--platform', choices=TEMPLATE_BUILDERS.keys(), required=True,
                            help='The platform of the LLM')
    llm_parser.add_argument('-m', '--model', required=True, help='The name of the model')
    llm_parser.add_argument('-i', '--instructions', required=True, help='Custom instructions for the model')
    llm_parser.add_argument('-f', '--filename', required=False, help='Configuration file name')

def main():
    parser = argparse.ArgumentParser(description="practice machine learning attacks from the command line")
    subparsers = parser.add_subparsers(dest='command', help='Type of action to make')

    save_template_parser = subparsers.add_parser('save_template', help='Save attack configuration templates')
    save_template_subparsers = save_template_parser.add_subparsers(dest='attack_type', help='types of attacks')

    add_parsers(save_template_subparsers)

    perform_action(parser.parse_args())

if __name__ == "__main__":
    main()