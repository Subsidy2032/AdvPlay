## AdvPlay

AdvPlay is a framework for running adversarial AI attacks with tunable parameters and reproducible results. Designed for red team assessments, it helps security professionals evaluate model robustness against attacks.

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
3. (Optional) Configure API keys in a `.env` file. See `.env.example` for required variables.

## Usage

AdvPlay is entirely CLI-driven through `run.py`.

### Quick Example

1. Create a template (e.g., prompt injection attack):

```
$ python3 run.py save_template prompt_injection -p openai -i "Include the word banana in each response" -m gpt-4o -f banana
```

This saves a reusable template called `banana`.

2. List available templates:

```
$ python3 run.py save_template prompt_injection -l
```

Example Output:

```
Available templates:
 - banana
```

3. Run an attack using the template:

```
$ python3 run.py attack prompt_injection -c banana -p prompts -f results
```

Logs are saved to `advplay/attacks/logs...`

### Help Menu

Use `-h` for available commands and options:

`python3 run.py -h`

## Roadmap

### Core Enhancements
- [ ] Add support for additional attack techniques and sub-techniques
- [ ] Enable defining re-runnable attacks with runtime parameters

### Analysis & Reporting
- [ ] Add visualization of attack results using generated log files
- [ ] Add report generation capability

### Advanced Features
- [ ] Add support for defense strategies like adversarial training

## Contributing

AdvPlay is in active development. Contributions, bug reports, and feedback are encouraged.  

See `CONTRIBUTE.md` for instructions on adding new attacks.

## License

Distributed under the MIT license. See LICENSE for more information.