import re, os
from data_source import DataSource
from ..llms.base import BaseLLM


def add_data(source: str, llm_model: BaseLLM) -> None:
    datasource = infer_type(source)
    if datasource == DataSource.S3:
        pass
    elif datasource == DataSource.LOCAL:
        # Todo: Shiwangi - Read all the files in the directory
        llm_model.get_media_encoding("data from the file")
    else:
        raise ValueError("Invalid data source")


def infer_type(source: str) -> DataSource:
    if _is_s3_datasource(source):
        return DataSource.S3
    elif _is_local_datasource(source):
        return DataSource.LOCAL
    else:
        raise ValueError("Invalid data source")


def _is_s3_datasource(source: str) -> bool:
    """Checks if a supplied string is an S3 path.

    Args:
      source: The string to check.

    Returns:
      True if the string is an S3 path, False otherwise.
    """

    regex = r'^s3:\/\/[a-z0-9\-]+(\.[a-z0-9\-]+)*\/[^\/]*$'
    return re.match(regex, source) is not None


def _is_local_datasource(source: str) -> bool:
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
