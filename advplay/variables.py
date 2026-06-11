class commands:
    COMMAND: str = 'command'

    SAVE_TEMPLATE: str = 'save_template'
    ATTACK: str = 'attack'
    VISUALIZE: str = 'visualize'

    ATTACK_TYPE: str = 'attack_type'
    TECHNIQUE: str = 'technique'

class available_platforms():
    OPENAI: str = 'openai'

class available_frameworks():
    SKLEARN: str = 'sklearn'
    PYTORCH: str = 'pytorch'

class available_attacks():
    PROMPT_INJECTION: str = 'prompt_injection'
    POISONING: str = 'poisoning'
    EVASION: str = 'evasion'

class poisoning_techniques():
    LABEL_FLIPPING: str = 'label_flipping'
    CLEAN_LABEL: str = 'clean_label'
    BACKDOOR: str = 'backdoor'

class evasion_techniques():
    FGSM: str = 'fgsm'
    BIM: str = 'bim'
    JSMA: str = 'jsma'
    C_W: str = 'c_w'
    PGD: str = 'pgd'
    GOODWORDS: str = 'goodwords'
    DEEPFOOL: str = 'deepfool'

class prompt_injection_techniques():
    DIRECT: str = 'direct'

class available_models():
    LOGISTIC_REGRESSION: str = 'logistic_regression'
    ONE_VS_REST: str = 'one_vs_rest'
    NAIVE_BAYES: str = 'naive_bayes'

    CNN: str = 'cnn'
    MNIST_CNN: str = 'mnist_cnn'
    CIFAR10_CNN: str = 'cifar10_cnn'
    SIMPLE_CLASSIFIER: str = 'simple_classifier'

class default_template_file_names():
    CUSTOM_INSTRUCTIONS: str = 'custom_instructions'
    LABEL_FLIPPING: str = 'label_flipping'
    POISONING_ATTACK_TEMPLATE: str = 'poisoning_attack_template'
    EVASION_ATTACK_TEMPLATE: str = 'evasion_attack_template'

class dataset_formats():
    CSV: str = 'csv'
    NPZ: str = 'npz'
    NPY: str = 'npy'

class locations():
    JSON: str = 'json'
