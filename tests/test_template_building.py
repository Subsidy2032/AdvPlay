from dotenv import load_dotenv
load_dotenv()

from advplay.attack_templates import template_registery
from advplay.attack_templates import template_builders
from advplay.paths import *

import unittest
import json
import os

class TestTemplateBuilders(unittest.TestCase):
    def test_template_building_with_file_name(self):
        expected_json = {
            "model": "gpt-4o",
            "instructions": "Those are the custom instructions instructions"
        }

        model = 'gpt-4o'
        instructions = 'Those are the custom instructions instructions'
        filename = 'new_custom_instructions'
        template_registery.define_template('openai', model=model, instructions=instructions, filename=filename)

        file_path = OPENAI_TEMPLATES / f"{filename}.json"
        self.assertTrue(file_path.exists(), f"{file_path} was not created")
        with open(file_path, 'r') as f:
            json_file = json.load(f)

        self.assertEqual(expected_json, json_file)
        os.remove(OPENAI_TEMPLATES / f"{filename}.json")

if __name__ == '__main__':
    unittest.main()