## Extending AdvPlay

AdvPlay is built around **self-registering base classes**. To plug in something new, you subclass the right base, pass a key to `__init_subclass__`, and drop the file anywhere under `advplay/`. On startup `load_required_classes` walks the tree and imports everything, which triggers registration automatically — no manual wiring, no CLI changes.

That's the whole trick. The rest of this doc shows what it looks like for three common extensions. Other base classes (`BaseModelLoader`, `BaseDatasetLoader`, `BaseModelSaver`, `BaseDatasetSaver`, `BaseTrainer`, `BaseEvaluator`, `BasePlatform`, ...) follow the exact same pattern.

### Add a new attack

Subclass `BaseAttack` (or an existing attack class for a new technique) and register it with `attack_type` / `attack_subtype`. Declare parameters with `Annotated[...]` — they automatically become CLI flags.

```python
from typing import Annotated
from advplay.attacks.base_attack import BaseAttack
from advplay.attacks.attack_param import AttackParam, TemplateParam

class MyAttack(BaseAttack, attack_type="my_attack", attack_subtype="naive"):
    target: Annotated[str, TemplateParam(type=str, required=True, default=None, help="Target to attack")]
    payload: Annotated[str, AttackParam(type=str, required=True, default=None, help="Payload to send")]

    def execute(self):
        # returns (results, artifacts, extras) — see existing attacks for the exact shape
        ...
```

Save it under `advplay/attacks/my_attack/` and it's instantly available as `python3 run.py attack my_attack naive --payload ...`.

### Add an attack evaluator

Subclass `BaseAttackEvaluator` and bind it with `attack_type` (and optionally `attack_subtype`). The orchestrator looks up `(attack_type, attack_subtype)` and falls back to `(attack_type, None)` when no technique-specific evaluator is registered.

```python
from advplay.attack_evaluators.base_attack_evaluator import BaseAttackEvaluator

class MyAttackEvaluator(BaseAttackEvaluator, attack_type="my_attack", attack_subtype=None):
    def evaluate(self, context):
        ...
        return results, artifacts, extras

class MyAttackNaiveEvaluator(BaseAttackEvaluator, attack_type="my_attack", attack_subtype="naive"):
    def evaluate(self, context):
        ...
```

Define a matching `BaseEvaluationContext` subclass under `attack_evaluators/contexts/` so the orchestrator knows what to pass in.

### Add a visualizer

Same idea — subclass `BaseVisualizer`, bind with `attack_type` (and optionally `attack_subtype`), and render whatever you want from the context (plots, tables, dashboards). Technique-specific visualizers override the default; if none is registered for a technique, the `(attack_type, None)` visualizer is used.

```python
from advplay.visualization.base_visualizer import BaseVisualizer

class MyAttackVisualizer(BaseVisualizer, attack_type="my_attack", attack_subtype=None):
    def visualize(self, context):
        ...

class MyAttackNaiveVisualizer(BaseVisualizer, attack_type="my_attack", attack_subtype="naive"):
    def visualize(self, context):
        ...
```

Outputs go under the `outputs` directory.

### Adding other component types

The same recipe extends to model/dataset **loaders** and **savers**, trainers, evaluators, LLM platforms, loss functions, optimizers, and architectures. Find the matching `base_*.py` under `advplay/ml/`, subclass it with the appropriate registration key (e.g. `source_type="parquet"` for a dataset loader), and the framework will pick it up on the next run.

### Rule of thumb

If you find yourself editing the CLI or a big `if/elif` switch to make your class work — you're fighting the framework. Find the relevant base class, subclass it, and let the registry do the rest.