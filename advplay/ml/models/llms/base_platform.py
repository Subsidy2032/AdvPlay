from advplay import paths

class BasePlatform:
    registry = {}

    def __init_subclass__(cls, platform: str):
        if platform in BasePlatform.registry:
            raise ValueError(f"LLM platform already registered: {platform}")

        super().__init_subclass__()
        BasePlatform.registry[platform] = cls

    def __init__(self, model, instructions, session_id):
        self.model = model
        self.instructions = instructions
        self.session_id = session_id
        self.conversation = None

    def query_llm(self, prompt):
        raise NotImplementedError("Subclasses must implement the query method.")

    def clear_chat_history(self):
        raise NotImplementedError("Subclasses must implement the get_conversation_history method.")

    def get_conversation_history(self):
        raise NotImplementedError("Subclasses must implement the get_conversation_history method.")

