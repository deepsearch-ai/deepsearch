from typing import List, Sequence, Union

Vector = Union[Sequence[float], Sequence[int]]
Embedding = Vector
Embeddings = List[Embedding]
Document = str
Documents = List[Document]


class EmbeddingFunction:
    def __call__(self, texts: Documents) -> Embeddings:
        ...


class BaseVectorDatabaseConfig:
    def __init__(self):
        pass
