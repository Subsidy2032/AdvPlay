from langchain_openai import ChatOpenAI
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.chat_history import InMemoryChatMessageHistory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.schema import HumanMessage, AIMessage
from datetime import datetime
import json
import os
from openai import OpenAI

from advplay.utils.append_log_entry import append_log_entry
from advplay.attacks.prompt_injection.prompt_injection_attack import PromptInjectionAttack
from advplay.variables import available_platforms, available_attacks

class OpenAIPromptInjectionAttack(PromptInjectionAttack, attack_type=available_attacks.PROMPT_INJECTION, attack_subtype=available_platforms.OPENAI):
    def __init__(self, template, **kwargs):
        super().__init__(template, **kwargs)

    def execute(self):
        super().execute()

        openai_key = os.getenv("OPENAI_API_KEY")
        openrouter_key = os.getenv("OPENROUTER_API_KEY")
        openrouter_url = os.getenv("OPENROUTER_BASE_URL")

        if openai_key:
            chat = ChatOpenAI(model=self.model)
        elif openrouter_key and openrouter_url:
            chat = ChatOpenAI(
                model=self.model,
                api_key=openrouter_key,
                base_url=openrouter_url
            )
        else:
            raise ValueError("No valid API key found for OpenAI or OpenRouter.")

        prompt = ChatPromptTemplate.from_messages([
            ("system", self.custom_instructions),
            MessagesPlaceholder(variable_name="history"),
            ("human", "{input}")
        ])

        session_histories = {}

        def get_session_history(session_id: str):
            if session_id not in session_histories:
                session_histories[session_id] = InMemoryChatMessageHistory()
            return session_histories[session_id]

        conversation = RunnableWithMessageHistory(
            prompt | chat,
            get_session_history=get_session_history,
            input_messages_key="input",
            history_messages_key="history"
        )

        session_id = self.session_id

        if self.prompt_list and (len(self.prompt_list) > 0):
            for prompt in self.prompt_list:
                response = conversation.invoke(
                    {"input": prompt},
                    config={"configurable": {"session_id": session_id}}
                )

                print(f"Trying prompt: {prompt}")
                print(f"Response: {response.content}\n")

            self.log_chat_history(session_histories, self.log_file_path)
            print(f"Chat history saved to {self.log_file_path}. Exiting...")
            return

        print("Start trying different prompts. Type 'clear' to clear conversation history, and 'exit' to exit")

        while True:
            user_input = input("> ")

            if user_input.lower() == "clear":
                session_histories[session_id].clear()
                print("Chat history cleared!")
                continue

            if user_input.lower() == "exit":
                self.log_chat_history(session_histories, self.log_file_path)
                print(f"Chat history saved to {self.log_file_path}. Exiting...")
                break

            response = conversation.invoke(
                {"input": user_input},
                config={"configurable": {"session_id": session_id}}
            )

            print(f"Response: {response.content}")

    def log_chat_history(self, history_obj, log_file_path):
        session_id, chat_history = next(iter(history_obj.items()))
        messages = chat_history.messages

        conversation = []

        for i in range(0, len(messages), 2):
            human = messages[i]
            ai = messages[i + 1] if i + 1 < len(messages) else None

            if isinstance(human, HumanMessage):
                prompt = human.content
            else:
                continue

            response = ""
            if isinstance(ai, AIMessage):
                response = ai.content

            conversation.append({
                "prompt": prompt,
                "response": response
            })

        log_entry = {
            "attack": self.attack_type,
            "technique": self.technique,
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "session_id": session_id,
            "model": self.model,
            "instructions": self.custom_instructions,
            "total_turns": len(conversation),
            "conversation": conversation
        }

        append_log_entry(log_file_path, log_entry)

    def build(self):
        try:
            client = OpenAI()
            models = client.models.list()
            model_names = [model.id for model in models.data]

        except Exception as e:
            print(e)
            model_names = []

        if self.model not in model_names:
            raise NameError(f"An OpenAI model with the name {self.model} does not exist. "
                            f"Some popular OpenAI models are gpt-5 and gpt-5-mini.")

        super().build()