from langchain_openai import ChatOpenAI
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.chat_history import InMemoryChatMessageHistory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.schema import HumanMessage, AIMessage
from datetime import datetime
import json

class OpenAIPromptInjectionAttack():
    def __init__(self, model: str, instructions: str, session_id: str, log_file_path: str):
        self.model = model
        self.instructions = instructions
        self.session_id = session_id
        self.log_file_path = log_file_path

    def execute(self):
        chat = ChatOpenAI(model=self.model)

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

        session_id = self.session_id
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

            print(f"AI: {response.content}")

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
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "session_id": session_id,
            "model": self.model,
            "instructions": self.instructions,
            "total_turns": len(conversation),
            "conversation": conversation
        }

        with open(log_file_path, 'w', encoding='utf-8') as f:
            json.dump(log_entry, f, ensure_ascii=False, indent=2)
