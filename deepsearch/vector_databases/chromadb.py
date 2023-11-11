import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

from langchain.document_loaders import GoogleApiClient, GoogleApiYoutubeLoader

from ..enums import MEDIA_TYPE
from ..types import MediaData
from .base import BaseVectorDatabase
from .configs.chromadb import ChromaDbConfig

google_api_client = GoogleApiClient(credentials_path=Path("your_path_creds.json"))

# Use a Channel
youtube_loader_channel = GoogleApiYoutubeLoader(
    google_api_client=google_api_client,
    channel_name="Reducible",
    captions_language="en",
)

youtube_loader_channel.load()

try:
    import chromadb
    from chromadb import Collection, QueryResult
    from chromadb.config import Settings
    from chromadb.errors import InvalidDimensionException

except ImportError:
    raise ImportError("Chromadb requires extra dependencies") from None


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
        self._get_or_create_collection(
            self.config.audio_collection_name,
            self.config.image_collection_name,
            self.config.video_collection_name,
        )
        super().__init__(config=self.config)

    def add(
        self,
        embeddings: List[List[float]],
        documents: List[str],
        ids: List[str],
        metadata: List[Any],
        media_type: MEDIA_TYPE,
    ) -> List[str]:
        size = len(documents)
        if embeddings is not None and len(embeddings) != size:
            raise ValueError(
                "Cannot add documents to chromadb with inconsistent embeddings"
            )
        collection = self.collections[media_type]
        # embedding would be created by the llm model used
        for i in range(0, len(documents), self.BATCH_SIZE):
            print(
                "Inserting batches from {} to {} in chromadb".format(
                    i, min(len(documents), i + self.BATCH_SIZE)
                )
            )
            if embeddings is not None:
                collection.add(
                    embeddings=embeddings[i : i + self.BATCH_SIZE],
                    documents=documents[i : i + self.BATCH_SIZE],
                    ids=ids[i : i + self.BATCH_SIZE],
                    metadatas=metadata[i : i + self.BATCH_SIZE],
                )

            else:
                collection.add(
                    documents=documents[i : i + self.BATCH_SIZE],
                    ids=ids[i : i + self.BATCH_SIZE],
                    metadatas=metadata[i : i + self.BATCH_SIZE],
                )
        return []

    def query(
        self,
        input_query: str,
        input_embeddings: List[float],
        n_results: int,
        media_type: MEDIA_TYPE,
        distance_threshold: float,
    ) -> List[MediaData]:
        if input_embeddings:
            query_params = {
                "query_embeddings": [input_embeddings],
                "n_results": n_results,
            }
        else:
            query_params = {"query_texts": [input_query], "n_results": n_results}

        media_data = []

        collection = self.collections[media_type]
        try:
            results = collection.query(**query_params)
        except InvalidDimensionException as e:
            raise InvalidDimensionException(
                e.message()
                + ". This is commonly a side-effect when an embedding function, different from the one used to"
                " add the embeddings, is used to retrieve an embedding from the database."
            ) from None
        filtered_results = self.filter_query_result_by_distance(
            results, distance_threshold
        )
        if len(filtered_results.get("documents", [])) == 0:
            return media_data

        documents = filtered_results.get("documents")[0]
        metadatas = filtered_results.get("metadatas")[0]
        for document, metadata in zip(documents, metadatas):
            media_data.append({"document": document, "metadata": metadata})
        return media_data

    def filter_query_result_by_distance(
        self, query_result: QueryResult, distance_threshold: float
    ) -> QueryResult:
        filtered_result: QueryResult = {
            "ids": [],
            "embeddings": [],
            "documents": [],
            "metadatas": [],
            "distances": [],
        }

        for i, ids in enumerate(query_result["ids"]):
            filtered_subresult = {
                "ids": [],
                "embeddings": [],
                "documents": [],
                "metadatas": [],
                "distances": [],
            }
            if query_result["distances"][i] is None:
                continue

            for j, distance in enumerate(query_result["distances"][i]):
                if distance >= distance_threshold:
                    filtered_subresult["ids"].append(query_result["ids"][i][j])

                    if "embeddings" in query_result and query_result["embeddings"]:
                        embeddings = query_result["embeddings"][i]
                        filtered_subresult["embeddings"].append(embeddings[j])

                    if "documents" in query_result and query_result["documents"]:
                        documents = query_result["documents"][i]
                        filtered_subresult["documents"].append(documents[j])

                    if "metadatas" in query_result and query_result["metadatas"]:
                        metadatas = query_result["metadatas"][i]
                        filtered_subresult["metadatas"].append(metadatas[j])

                    filtered_subresult["distances"].append(distance)

            if filtered_subresult["ids"]:
                filtered_result["ids"].append(filtered_subresult["ids"])
                filtered_result["distances"].append(filtered_subresult["distances"])

                if filtered_subresult["embeddings"]:
                    filtered_result["embeddings"].append(
                        filtered_subresult["embeddings"]
                    )

                if filtered_subresult["documents"]:
                    filtered_result["documents"].append(filtered_subresult["documents"])

                if filtered_subresult["metadatas"]:
                    filtered_result["metadatas"].append(filtered_subresult["metadatas"])

        return filtered_result

    def get_existing_document_ids(
        self, metadata_filters, media_type: MEDIA_TYPE
    ) -> List[str]:
        query_args = {"where": self._generate_where_clause(metadata_filters)}
        collection = self.collections[media_type]

        results = []
        offset = 0
        first_iteration = True
        while offset != -1 or first_iteration:
            first_iteration = False
            query_result = collection.get(
                **query_args, offset=offset, limit=self.BATCH_SIZE
            )
            metadatas = query_result.get("metadatas", [])
            document_ids = list(
                map(lambda metadata: metadata.get("document_id", []), metadatas)
            )
            results.extend(document_ids)
            offset = offset + min(self.BATCH_SIZE, len(query_result.get("ids")))
            if len(query_result.get("ids")) == 0:
                break
        return results

    def count(self) -> Dict[str, int]:
        """
        Count number of documents/chunks embedded in the database.

        :return: number of documents
        """
        return {
            "image_collection": self.collections[MEDIA_TYPE.IMAGE].count(),
            "audio_collection": self.collections[MEDIA_TYPE.AUDIO].count(),
            "video_collection": self.collections[MEDIA_TYPE.VIDEO].count(),
        }

    def delete(self, where, media_type: Optional[MEDIA_TYPE] = None):
        if not media_type or media_type == MEDIA_TYPE.AUDIO:
            self.collections[MEDIA_TYPE.AUDIO].delete(where=where)
        if not media_type or media_type == MEDIA_TYPE.IMAGE:
            self.collections[MEDIA_TYPE.IMAGE].delete(where=where)
        if not media_type or media_type == MEDIA_TYPE.VIDEO:
            self.collections[MEDIA_TYPE.VIDEO].delete(where=where)

    def reset(self):
        """
        Resets the database. Deletes all embeddings irreversibly.
        """
        # Delete all data from the collection
        try:
            self.client.delete_collection(self.config.audio_collection_name)
            self.client.delete_collection(self.config.image_collection_name)
        except ValueError:
            raise ValueError(
                "For safety reasons, resetting is disabled. "
                "Please enable it by setting `allow_reset=True` in your ChromaDbConfig"
            ) from None
        # Recreate
        self._get_or_create_collection(
            self.config.audio_collection_name,
            self.config.image_collection_name,
            self.config.video_collection_name,
        )

    def _get_or_create_collection(
        self,
        audio_collection_name: str,
        image_collection_name: str,
        video_collection_name: str,
    ) -> None:
        audio_collection = self.client.get_or_create_collection(
            name=audio_collection_name,
            embedding_function=self.config.embedding_function,
            metadata={"hnsw:space": "cosine"},
        )
        image_collection = self.client.get_or_create_collection(
            name=image_collection_name,
            embedding_function=self.config.embedding_function,
            metadata={"hnsw:space": "cosine"},
        )
        video_collection = self.client.get_or_create_collection(
            name=video_collection_name,
            embedding_function=self.config.embedding_function,
            metadata={"hnsw:space": "cosine"},
        )
        self.collections = {
            MEDIA_TYPE.AUDIO: audio_collection,
            MEDIA_TYPE.IMAGE: image_collection,
            MEDIA_TYPE.VIDEO: video_collection,
        }

    def _generate_where_clause(self, where_clause: Dict[str, any]):
        # If only one filter is supplied, return it as is
        # (no need to wrap in $and based on chroma docs)
        if not where_clause:
            return {}
        if len(where_clause.keys()) == 1:
            value = list(where_clause.values())[0]
            key = list(where_clause.keys())[0]
            if isinstance(value, list):
                where_filter = {key: {"$in": value}}
            else:
                where_filter = {key: value}
            return where_filter
        where_filters = []
        for k, v in where_clause.items():
            if isinstance(v, list):
                where_filters.append({k: {"$in": v}})
            if isinstance(v, str):
                where_filters.append({k: v})
        return {"$and": where_filters}
