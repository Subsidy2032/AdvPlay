# Contributing to AdvPlay

You can contribute by adding new attack techniques or sub-techniques, implementing new features, reporting bugs, or giving feedback.

---

## Adding a New Attack Technique

1. **Register the attack**

   Add a variable in the `available_attacks` class:

    ```python
   class available_attacks():
        PROMPT_INJECTION: str = 'prompt_injection'
        POISONING: str = 'poisoning'
    ```
   
2. **Define sub-types**

    Add a class describing available platforms/sub-types:

    ```python
   class available_platforms():
        OPENAI: str = 'openai'
    ```
   
3. **Add parsers**

    Define parameters in run.py using:

   - `add_save_template_[attack]_parser(save_template_parser)`
   - `add_attack_[attack]_parser(attack_parser)`

    Call these methods from `main()` in `run.py`.

4. **Add handlers**

Create a handler file under `advplay/command_dispatcher/handlers/`. You can take inspiration from available handlers.

5. **Create folders**

    - advplay/attack_templates/template_builders/[attack]
    - advplay/attacks/[attack]

6. **Implement the attack class**

Add a class under `advplay/attacks/[attack]` that sets up the infrastructure to run sub-techniques. This class does **not** execute the attack itself—it simply orchestrates sub-techniques when called.

### Adding a New Sub-technique

Once the attack infrastructure is in place, adding a sub-technique is simpler:

1. Add a template builder class under template_builders/[attack]. 
2. Add an Attack class under attacks/[attack].
3. Register the new sub-technique in the attack technique class by adding it to `platforms_cls`:

    ```python
    self.platforms_cls = {
        [available_sub_techniques].[sub_technique]: [SubTechniqueClass]
    }
    ```

Test the sub-technique to ensure it can be executed via run.py and logs results correctly.