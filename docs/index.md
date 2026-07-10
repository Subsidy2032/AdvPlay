# AdvPlay

AdvPlay is a framework for running adversarial AI attacks with tunable parameters and
reproducible results. Designed for red team assessments and research, it helps security
professionals evaluate model robustness against attacks — all from the command line.

!!! warning "Disclaimer"
    This tool is intended strictly for research, security testing, and red team
    assessments. Using it against systems, APIs, or models without explicit permission is
    illegal and unethical. By using this software you accept full responsibility for your
    actions; the developers take **no liability** for misuse or damage.

## Where to go next

- **[Get started](Get Started/Get Started.md)** — install AdvPlay and run your first
  attack (the Banana Challenge and a poisoning walkthrough).
- **[CLI reference](cli/index.md)** — every command, attack, and parameter, generated
  directly from the attack registry.
- **[API reference](reference/index.md)** — the Python classes and functions, for anyone
  extending the framework.
- **[Extending AdvPlay](Extending AdvPlay/Extending AdvPlay.md)** — how the self-registering
  base classes work and how to add new attacks, evaluators, and visualizers.

## Supported attacks

| Attack | Techniques | Domain |
|---|---|---|
| `prompt_injection` | `direct` | LLM |
| `poisoning` | `label_flipping` | Classical ML |
| `evasion` | `fgsm`, `bim`, `jsma`, `c_w`, `pgd` | Classical ML / Deep Learning |

See the [CLI reference](cli/index.md) for the full, always-current list and every
parameter each attack accepts.
