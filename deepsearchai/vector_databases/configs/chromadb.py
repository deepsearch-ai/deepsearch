import logging
from typing import Optional

from .base import BaseVectorDatabaseConfig, EmbeddingFunction

try:
    import chromadb
    from chromadb.config import Settings
    from chromadb.errors import InvalidDimensionException
    from chromadb.utils.embedding_functions import \
        SentenceTransformerEmbeddingFunction

except RuntimeError:
    pass


class ChromaDbConfig(BaseVectorDatabaseConfig):
    def __init__(
        self,
        dir: Optional[str] = None,
        host: Optional[str] = None,
        port: Optional[str] = None,
        allow_reset=False,
        chroma_settings: Optional[dict] = None,
        embedding_function: Optional[EmbeddingFunction] = None,
    ):
        """
        Initializes a configuration class instance for ChromaDB.

        :param dir: Path to the database directory, where the database is stored, defaults to None
        :type dir: Optional[str], optional
        :param host: Database connection remote host. Use this if you run Embedchain as a client, defaults to None
        :type host: Optional[str], optional
        :param port: Database connection remote port. Use this if you run Embedchain as a client, defaults to None
        :type port: Optional[str], optional
        :param allow_reset: Resets the database. defaults to False
        :type allow_reset: bool
        :param chroma_settings: Chroma settings dict, defaults to None
        :type chroma_settings: Optional[dict], optional
        """

        self.settings = Settings()
        self.settings.allow_reset = allow_reset
        self.embedding_function = (
            SentenceTransformerEmbeddingFunction()
            if not embedding_function
            else embedding_function
        )
        if chroma_settings:
            for key, value in chroma_settings.items():
                if hasattr(self.settings, key):
                    setattr(self.settings, key, value)

        if host and port:
            logging.info(f"Connecting to ChromaDB server: {host}:{port}")
            self.settings.chroma_server_host = host
            self.settings.chroma_server_http_port = port
            self.settings.chroma_api_impl = "chromadb.api.fastapi.FastAPI"
        else:
            if dir is None:
                dir = "db"

            self.settings.persist_directory = dir
            self.settings.is_persistent = True

        super().__init__()
