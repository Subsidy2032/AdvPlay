### Prerequisites

- Python 3
- pip

### Steps

1. Create and activate a virtual environment:    
    ```
    python3 -m venv .venv
    source .venv/bin/activate   # On Windows: .venv\Scripts\activate
    ```
2. Clone the repository:
    ```
    git clone https://github.com/Subsidy2032/AdvPlay.git
    cd AdvPlay
    ```
3. Install dependencies:    
    ```
    pip install -r requirements.txt
    ```
4. (Optional) Configure environment variables:
    - Copy `.env.example` to `.env`
    - Add API keys if required by specific features
5. Verify installation:
    ```
    python3 run.py -h
    ```
    This should display the CLI help message without errors.

### Examples

#### Prompt Injection

A quick, interactive way to feel how AdvPlay works. We'll instruct the model to *never* say "banana", and then you'll try to trick it into saying it anyway.

1. Create a template that tells the model to keep its mouth shut about bananas, you can change the model that you want to test:
    ```
    python3 run.py save_template prompt_injection --platform openai --model gpt-4o-mini --custom-instructions "Never say banana" --template-filename banana
    ```
2. Launch an interactive prompt injection attack against it:
    ```
    python3 run.py attack prompt_injection direct --template banana
    ```
3. The CLI drops you into a chat. Try things like role-play, translations, encodings, spelling tricks — whatever it takes to get the word "banana" out of the model. Type `clear` to reset the conversation, `exit` when you're done.

Your full conversation is logged under `outputs/logs/prompt_injection/` so you can revisit successful (and unsuccessful) attempts later.

#### Poisoning
1. Create a template:
	```
	python3 run.py save_template poisoning --training-framework sklearn --model one_vs_rest --test-portion 0.2 --min-portion-to-poison 0.4 --max-portion-to-poison 0.8 --template-filename poisoning_attack_template
	```
2. Run the attack:
	```
	python3 run.py attack poisoning label_flipping --template poisoning_attack_template --dataset WineQT.csv --label-column quality --model-name first_poison	
	```