from datetime import datetime

from advplay.utils.append_log_entry import append_log_entry
from advplay.attacks.prompt_injection.prompt_injection_attack import PromptInjectionAttack
from advplay.variables import available_attacks, prompt_injection_techniques
from advplay.model_ops.llms.base_platform import BasePlatform

class DirectPromptInjectionAttack(PromptInjectionAttack, attack_type=available_attacks.PROMPT_INJECTION,
                                  attack_subtype=prompt_injection_techniques.DIRECT):
    def execute(self):
        super().execute()

        llm_cls = BasePlatform.registry.get(self.platform)
        llm = llm_cls(self.model, self.custom_instructions, self.session_id)

        if self.prompt_list and (len(self.prompt_list) > 0):
            for prompt in self.prompt_list:
                response = llm.query_llm(prompt)

                print(f"Trying prompt: {prompt}")
                print(f"Response: {response.content}\n")

            conversation_history = llm.get_conversation_history()
            self.log_chat_history(conversation_history, self.log_file_path)
            print(f"Chat history saved to {self.log_file_path}. Exiting...")
            return

        print("Start trying different prompts. Type 'clear' to clear conversation history, and 'exit' to exit")

        c = 0
        while True:
            user_input = input("> ")

            if user_input.lower() == "clear":
                llm.clear_chat_history()
                print("Chat history cleared!")
                continue

            if user_input.lower() == "exit":
                if c == 0:
                    return

                conversation_history = llm.get_conversation_history()
                self.log_chat_history(conversation_history, self.log_file_path)
                print(f"Chat history saved to {self.log_file_path}. Exiting...")
                break

            response = llm.query_llm(user_input)
            print(f"Response: {response.content}")
            c += 1

    def log_chat_history(self, conversation_history, log_file_path):
        log_entry = {
            "attack": self.attack_type,
            "technique": self.attack_subtype,
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "session_id": self.session_id,
            "model": self.model,
            "instructions": self.custom_instructions,
            "total_turns": len(conversation_history),
            "conversation": conversation_history
        }

        append_log_entry(log_file_path, log_entry)

    def build(self):
        llm = BasePlatform.registry.get(self.platform)
        super().build()