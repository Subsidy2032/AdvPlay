## AdvPlay

AdvPlay is a framework for running adversarial AI attacks with tunable parameters and reproducible results. Designed for red team assessments and research purposes, it helps security professionals evaluate model robustness against attacks.

## Disclaimer

This tool is intended strictly for research, security testing, and red team assessments. Using it against systems, APIs, or models without explicit permission is illegal and unethical.  

By using this software, you accept full responsibility for your actions. The developers take **no liability** for misuse, damage, or chaos caused by experiments conducted with this tool.  

## Getting Started

Tested on Python 3.11.13. Follow these steps to get AdvPlay running:

### Installation

1. Clone the repo:

    `git clone https://github.com/Subsidy2032/AdvPlay.git`

2. Install dependencies:

    `pip install -r requirements.txt`

3. (Optional) Configure API keys in a `.env` file. See `.env.example` for variable names.

## Usage

AdvPlay is entirely CLI-driven through `run.py`.

### Quick Example — The Banana Challenge 🍌

Tell a model to *never* say "banana", then try to make it say "banana" anyway.

1. Create a template that instructs the model to stay away from the B-word:

```
$ python3 run.py save_template prompt_injection --platform openai --model gpt-4o-mini --custom-instructions "Never say banana" --template-filename banana
```

This saves a reusable template called `banana`.

2. List available templates:

```
$ python3 run.py save_template prompt_injection --list
```

Example Output:

```
Available templates:
 - banana
```

3. Run the attack in interactive mode and give it your best shot:

```
$ python3 run.py attack prompt_injection direct --template banana
```

Role-play, translations, encodings, typos, emotional appeals — anything goes. Type `clear` to reset the chat, `exit` when you're done.

Full conversations (successful or not) are saved under `outputs/logs/<attack_type>/` for later review.

### Supported Attacks and Techniques

| Attack | Techniques | Domain |
|---|---|---|
| `prompt_injection` | `direct` | LLM |
| `poisoning` | `label_flipping` | Classical ML |
| `evasion` | `fgsm`, `bim`, `jsma`, `c_w`, `pgd` | Classical ML / Deep Learning |

New attacks and techniques can be added as self-registering classes — see `docs/Extending AdvPlay/Extending AdvPlay.md`.

### Help Menu

Use `-h` for available commands and options:

`python3 run.py -h`

## Roadmap

### Core Enhancements
- [ ] Add support for additional attack techniques and sub-techniques
- [ ] Enable defining re-runnable attacks with runtime parameters

### Analysis & Reporting
- [x] Add visualization of attack results using generated log files
- [ ] Add report generation capability

### Advanced Features
- [ ] Add support for defense strategies like adversarial training

## Contributing

AdvPlay is in active development. Contributions, bug reports, and feedback are encouraged.  

See `CONTRIBUTE.md` for instructions on how to contribute.

## License

Distributed under the MIT license. See LICENSE for more information.