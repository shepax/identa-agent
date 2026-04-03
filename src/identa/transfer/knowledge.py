from identa.transfer.types import TransferableKnowledge

class KnowledgeStore:
    def __init__(self, path: str):
        self.path = path
        
    def get(self, key: str) -> TransferableKnowledge | None:
        return None
        
    def put(self, key: str, knowledge: TransferableKnowledge) -> None:
        pass
