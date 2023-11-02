import logging
from typing import Any, Dict, List, Optional, Tuple, Union

from .base import BaseVectorDatabase
from .configs.chromadb import ChromaDbConfig

try:
    import chromadb
    from chromadb.config import Settings
    from chromadb.errors import InvalidDimensionException
    from chromadb import Collection, QueryResult
except ImportError:
    raise ImportError(
        "Chromadb requires extra dependencies") from None


class ChromaDB(BaseVectorDatabase):
    """Vector database using ChromaDB."""

    BATCH_SIZE = 100

    def __init__(self, config: Optional[ChromaDbConfig] = None):
        """Initialize a new ChromaDB instance

        :param config: Configuration options for Chroma, defaults to None
        :type config: Optional[ChromaDbConfig], optional
        """
        if config:
            self.config = config
        else:
            self.config = ChromaDbConfig()

        self.client = chromadb.Client(self.config.settings)
        self._get_or_create_collection(self.config.collection_name)
        super().__init__(config=self.config)

    def add(self, embeddings: List[List[float]], documents: List[str], ids: List[str]) -> List[str]:
        size = len(documents)
        if embeddings is not None and len(embeddings) != size:
            raise ValueError("Cannot add documents to chromadb with inconsistent embeddings")

        for i in range(0, len(documents), self.BATCH_SIZE):
            print("Inserting batches from {} to {} in chromadb".format(i, min(len(documents), i + self.BATCH_SIZE)))
            if embeddings is not None:
                self.collection.add(
                    embeddings=embeddings[i: i + self.BATCH_SIZE],
                    documents=documents[i: i + self.BATCH_SIZE],
                    ids=ids[i: i + self.BATCH_SIZE],
                )
            else:
                self.collection.add(
                    documents=documents[i: i + self.BATCH_SIZE],
                    ids=ids[i: i + self.BATCH_SIZE],
                )
        return []

    def query(self, input_query: Union[List[float], str], n_results: int) -> List[str]:
        try:
            if type(input_query) == list:
                results = self.collection.query(
                    query_embeddings=[
                        input_query,
                    ],
                    n_results=n_results,
                )
            else:
                results = self.collection.query(
                    query_texts=[
                        input_query,
                    ],
                    n_results=n_results,
                )
        except InvalidDimensionException as e:
            raise InvalidDimensionException(
                e.message()
                + ". This is commonly a side-effect when an embedding function, different from the one used to add the"
                  " embeddings, is used to retrieve an embedding from the database."
            ) from None

        documents_set = set()
        for result in results.get("documents", []):
            if not result:
                continue
            documents_set.add(result[0])
        return list(documents_set)

    def get_existing_object_identifiers(self, object_identifiers) -> List[str]:
        args = {}
        if object_identifiers:
            args["ids"] = object_identifiers

        results = []
        offset = 0
        first_iteration = True
        while offset != -1 or first_iteration:
            first_iteration = False
            query_result = self.collection.get(**args, offset=offset, limit=self.BATCH_SIZE)
            results.extend(query_result.get("ids"))
            offset = offset + min(self.BATCH_SIZE, len(query_result.get("ids")))
            if len(query_result.get("ids")) == 0:
                break
        return results

    def count(self) -> int:
        """
        Count number of documents/chunks embedded in the database.

        :return: number of documents
        :rtype: int
        """
        return self.collection.count()

    def delete(self, where):
        return self.collection.delete(where=where)

    def reset(self):
        """
        Resets the database. Deletes all embeddings irreversibly.
        """
        # Delete all data from the collection
        try:
            self.client.delete_collection(self.config.collection_name)
        except ValueError:
            raise ValueError(
                "For safety reasons, resetting is disabled. "
                "Please enable it by setting `allow_reset=True` in your ChromaDbConfig"
            ) from None
        # Recreate
        self._get_or_create_collection(self.config.collection_name)

    def _get_or_create_collection(self, name: str) -> Collection:
        self.collection = self.client.get_or_create_collection(
            name=name,
            embedding_function=self.config.embedding_function,
        )
        return self.collection
