from dotenv import load_dotenv
load_dotenv()

from advplay.attack_templates.template_registery import TEMPLATE_BUILDERS
from advplay.attack_templates import template_builders
from advplay.main import perform_action
from advplay.variables import *

import argparse

def add_save_template_parsers(parser):
    llm_parser = parser.add_parser(parser_names.LLM, help='Perform attacks on LLM')
    llm_parser.add_argument('-p', '--platform', choices=TEMPLATE_BUILDERS.keys(), required=True,
                            help='The platform of the LLM')
    llm_parser.add_argument('-m', '--model', required=True, help='The name of the model')
    llm_parser.add_argument('-i', '--instructions', required=True, help='Custom instructions for the model')
    llm_parser.add_argument('-f', '--filename', required=False, help='Configuration file name')

def main():
    parser = argparse.ArgumentParser(description="practice machine learning attacks from the command line")
    subparsers = parser.add_subparsers(dest=parser_names.COMMAND, help='Type of action to make')

    save_template_parser = subparsers.add_parser(parser_names.SAVE_TEMPLATE, help='Save attack configuration templates')
    save_template_subparsers = save_template_parser.add_subparsers(dest=parser_names.ATTACK_TYPE, help='types of attacks')
    add_save_template_parsers(save_template_subparsers)

    perform_action(parser.parse_args())

if __name__ == "__main__":
    main()