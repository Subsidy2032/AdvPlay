class commands:
    COMMAND: str = 'command'

    SAVE_TEMPLATE: str = 'save_template'
    ATTACK: str = 'attack'

    ATTACK_TYPE: str = 'attack_type'

class available_platforms():
    OPENAI: str = 'openai'

class available_attacks():
    PROMPT_INJECTION: str = 'prompt_injection'

class available_training_algorithms():
    LOGISTIC_REGRESSION = 'logistic_regression'

class default_template_file_names():
    CUSTOM_INSTRUCTIONS: str = 'custom_instructions'