import mimetypes
import os
import re
from typing import Dict, List

from deepsearchai.embedding_models_config import EmbeddingModelsConfig
from deepsearchai.enums import MEDIA_TYPE
from deepsearchai.types import MediaData
from deepsearchai.vector_databases.base import BaseVectorDatabase
from .data_source import DataSource
from .local import LocalDataSource
from .s3 import S3DataSource
from .youtube import YoutubeDatasource


class SourceUtils:
    def __init__(self):
        self.local_data_source = LocalDataSource()
        self.s3_data_source = S3DataSource()
        self.youtube_data_source = YoutubeDatasource()

    def add_data(
        self,
        source: str,
        embedding_models_config: EmbeddingModelsConfig,
        vector_database: BaseVectorDatabase,
    ) -> None:
        datasource = self._infer_type(source)
        if datasource == DataSource.S3:
            self.s3_data_source.add_data(
                source, embedding_models_config, vector_database
            )
        elif datasource == DataSource.LOCAL:
            self.local_data_source.add_data(
                source, embedding_models_config, vector_database
            )
        elif datasource == DataSource.YOUTUBE:
            self.youtube_data_source.add_data(
                source, embedding_models_config, vector_database
            )
        else:
            raise ValueError("Invalid data source")

    def get_data(
        self,
        query: str,
        media_types: List[MEDIA_TYPE],
        embedding_models_config: EmbeddingModelsConfig,
        vector_database: BaseVectorDatabase,
    ) -> Dict[MEDIA_TYPE, List[MediaData]]:
        media_data = {}
        for media_type in media_types:
            if media_type == MEDIA_TYPE.UNKNOWN:
                continue
            media_data[media_type] = []
            for embedding_model in embedding_models_config.get_embedding_model(
                media_type
            ):
                media_data[media_type].extend(
                    vector_database.query(query, 1, media_type, 0.5, embedding_model)
                )
        return media_data

    def _infer_type(self, source: str) -> DataSource:
        if self._is_s3_path(source):
            return DataSource.S3
        elif self._is_local_datasource(source):
            return DataSource.LOCAL
        elif self._is_youtube_datasource(source):
            return DataSource.YOUTUBE
        else:
            raise ValueError("Invalid data source")

    def _is_s3_path(self, path: str):
        """Checks if a supplied string is an S3 path."""

        # Regex pattern for an S3 path
        # s3_path_regex = r'^s3://(?P<bucket>[A-Za-z0-9\-\.]+)/(?P<key>.*)$'
        s3_path_regex = r"^s3://(?P<bucket>[A-Za-z0-9\-\.\/]+)$"

        # Match the path against the regex pattern
        match = re.match(s3_path_regex, path)

        # If the path matches the regex pattern, then it is an S3 path
        return match is not None

    def _is_local_datasource(self, source: str) -> bool:
        """Checks if a supplied string is a local directory or a file.

        Args:
          source: The string to check.

        Returns:
          True if the string is a local directory or a file, False otherwise.
        """
        if os.path.isdir(source):
            return True
        elif os.path.isfile(source):
            return True
        else:
            return False

    def _is_youtube_datasource(self, source: str) -> bool:
        """Checks if a supplied string is a youtube channel id

        Args:
          source: The string to check.

        Returns:
          True if the string is a Youtube channel id
        """
        tokens = source.split(":")
        if len(tokens) == 2 and tokens[0] == "youtube":
            return True
        else:
            return False
