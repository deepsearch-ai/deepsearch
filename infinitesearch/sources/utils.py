import re, os
from .data_source import DataSource
from ..llms.base import BaseLLM
from ..vector_databases.base import BaseVectorDatabase
from PIL import Image, UnidentifiedImageError
from .local import LocalDataSource
from .s3 import S3DataSource


class SourceUtils:
    def __init__(self):
        self.local_data_source = LocalDataSource()
        self.s3_data_source = S3DataSource()

    def add_data(self, source: str, llm_model: BaseLLM, vector_database: BaseVectorDatabase) -> None:
        datasource = self._infer_type(source)
        if datasource == DataSource.S3:
            self.s3_data_source.add_data(source, llm_model, vector_database)
        elif datasource == DataSource.LOCAL:
            self.local_data_source.add_data(source, llm_model, vector_database)
        else:
            raise ValueError("Invalid data source")

    def query(self, query: str, llm_model: BaseLLM, vector_database: BaseVectorDatabase) -> None:
        encodings_json = llm_model.get_text_encoding(query)
        vector_database.query(encodings_json.get("embedding"), 1)

    def _infer_type(self, source: str) -> DataSource:
        if self._is_s3_datasource(source):
            return DataSource.S3
        elif self._is_local_datasource(source):
            return DataSource.LOCAL
        else:
            raise ValueError("Invalid data source")

    def _is_s3_datasource(self, source: str) -> bool:
        """Checks if a supplied string is an S3 path.

        Args:
          source: The string to check.

        Returns:
          True if the string is an S3 path, False otherwise.
        """

        regex = r'^s3:\/\/[a-z0-9\-]+(\.[a-z0-9\-]+)*\/[^\/]*$'
        return re.match(regex, source) is not None

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
