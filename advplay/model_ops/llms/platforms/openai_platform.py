from langchain_openai import ChatOpenAI
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.chat_history import InMemoryChatMessageHistory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.schema import HumanMessage, AIMessage
from datetime import datetime
import json
import os
from openai import OpenAI

from advplay.model_ops.llms.base_platform import BasePlatform
from advplay.variables import available_platforms

class OpenAIPlatform(BasePlatform, platform=available_platforms.OPENAI):
    def __init__(self, model, instructions, session_id):
        super().__init__(model, instructions, session_id)
        self.validate_model()

        chat = self.set_up_chat()
        self.conversation, self.session_histories = self.set_up_conversation(chat)

    def query_llm(self, prompt):
        return self.conversation.invoke({"input": prompt}, config={"configurable": {"session_id": self.session_id}})

    def clear_chat_history(self):
        self.session_histories[self.session_id].clear()

    def get_conversation_history(self):
        session_id, chat_history = next(iter(self.session_histories.items()))
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

        return conversation

    def set_up_chat(self):
        openai_key = os.getenv("OPENAI_API_KEY")
        openrouter_key = os.getenv("OPENROUTER_API_KEY")
        openrouter_url = os.getenv("OPENROUTER_BASE_URL")

        if openai_key:
            return ChatOpenAI(model=self.model)
        elif openrouter_key and openrouter_url:
            return ChatOpenAI(
                model=self.model,
                api_key=openrouter_key,
                base_url=openrouter_url
            )
        else:
            raise ValueError("No valid API key found for OpenAI or OpenRouter.")

    def set_up_conversation(self, chat):
        prompt = ChatPromptTemplate.from_messages([
            ("system", self.instructions),
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

        return conversation, session_histories

    def validate_model(self):
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