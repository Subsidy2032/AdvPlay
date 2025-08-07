from openai import OpenAI

class OpenAIPromptInjectionAttack():
    def __init__(self, model: str, instructions: str, filename: str):
        self.model = model
        self.instructions = instructions
        self.filename = filename

    def execute(self):
        print(f"Executing OpenAI attack\n"
              f"Model: {self.model}\n"
              f"Instructions: {self.instructions}\n"
              f"Results file name: {self.filename}")