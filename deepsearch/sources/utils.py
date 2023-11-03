import os
import re
from typing import List
import mimetypes

from ..enums import MEDIA_TYPE
from ..llms_config import LlmsConfig
from ..vector_databases.base import BaseVectorDatabase
from .data_source import DataSource
from .local import LocalDataSource
from .s3 import S3DataSource
from ..enums import MEDIA_TYPE

class SourceUtils:
    def __init__(self):
        self.local_data_source = LocalDataSource()
        self.s3_data_source = S3DataSource()

    def add_data(
            self, source: str, llms_config: LlmsConfig, vector_database: BaseVectorDatabase
    ) -> None:
        datasource = self._infer_type(source)
        if datasource == DataSource.S3:
            self.s3_data_source.add_data(source, llms_config, vector_database)
        elif datasource == DataSource.LOCAL:
            self.local_data_source.add_data(source, llms_config, vector_database)
        else:
            raise ValueError("Invalid data source")

    def query(
            self, query: str, data_types: List[MEDIA_TYPE], llms_config: LlmsConfig, vector_database: BaseVectorDatabase
    ) -> List[str]:
        data = []
        for data_type in data_types:
            if data_type == MEDIA_TYPE.UNKNOWN:
                continue
            encodings_json = llms_config.get_llm_model(data_type).get_text_encoding(query)
            data.extend(
                vector_database.query(encodings_json.get("text"), encodings_json.get("embedding"), 1, data_type))
        return data

    def _infer_type(self, source: str) -> DataSource:
        if self._is_s3_path(source):
            return DataSource.S3
        elif self._is_local_datasource(source):
            return DataSource.LOCAL
        else:
            raise ValueError("Invalid data source")

    def _is_s3_path(self, source: str) -> bool:
        """Checks if the provided string is an S3 path.

        Args:
          path: The string to check.

        Returns:
          True if the string is an S3 path, False otherwise.
        """
        # Check if the string starts with `s3://`.
        if not source.startswith("s3://"):
            return False

        # Check if the string contains a bucket name.
        bucket_name_regex = r"[a-z0-9.-]+[a-z0-9]+"
        match = re.match(bucket_name_regex, source.split("/")[2])
        if not match:
            return False

        if len(source.split("/")) == 3:
            return True

        # Check if the string contains an object key.
        object_key_regex = r"^/?[a-zA-Z0-9/_\-]+[a-zA-Z0-9/_\-]$"
        match = re.match(object_key_regex, "/".join(source.split("/")[3:]))
        if not match:
            return False

        return True

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