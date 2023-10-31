from .base import BaseVectorDatabase
from typing import Any


class ChromaDB(BaseVectorDatabase):
    def __init__(self):
        self.data = []
        super().__init__()

    def add(self, data: Any):
        import pdb; pdb.set_trace()
        self.data.append(data)

    def query(self, query: str):
        import pdb; pdb.set_trace()
        return self.data
