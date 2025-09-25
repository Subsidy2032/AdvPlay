class commands:
    COMMAND: str = 'command'

    SAVE_TEMPLATE: str = 'save_template'
    ATTACK: str = 'attack'
    VISUALIZE: str = 'visualize'

    ATTACK_TYPE: str = 'attack_type'

class available_platforms():
    OPENAI: str = 'openai'

class available_frameworks():
    SKLEARN: str = 'sklearn'

class available_attacks():
    PROMPT_INJECTION: str = 'prompt_injection'
    POISONING: str = 'poisoning'

class poisoning_techniques():
    LABEL_FLIPPING: str = 'label_flipping'

class prompt_injection_techniques():
    DIRECT: str = 'direct'

class available_training_algorithms():
    LOGISTIC_REGRESSION: str = 'logistic_regression'
    ONE_VS_REST: str = 'one_vs_rest'

class default_template_file_names():
    CUSTOM_INSTRUCTIONS: str = 'custom_instructions'
    LABEL_FLIPPING: str = 'label_flipping'

class dataset_formats():
    CSV: str = 'csv'
    NPZ: str = 'npz'
    NPY: str = 'npy'
