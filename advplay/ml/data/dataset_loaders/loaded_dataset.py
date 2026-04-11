class LoadedDataset:
    def __init__(self, data, source_type: str, metadata: dict = None):
        self.data = data
        self.source_type = source_type
        self.metadata = metadata or {}
