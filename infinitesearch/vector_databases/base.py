from typing import Any


class BaseVectorDatabase:
    def __init__(self):
        pass

    def add(self, data: Any):
        raise NotImplementedError

    def query(self, query: str):
        raise NotImplementedError
