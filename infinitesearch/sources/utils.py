import re, os
from .data_source import DataSource
from ..llms.base import BaseLLM
from ..vector_databases.base import BaseVectorDatabase
from PIL import Image, UnidentifiedImageError


def add_data(source: str, llm_model: BaseLLM, vector_database: BaseVectorDatabase) -> None:
    datasource = _infer_type(source)
    if datasource == DataSource.S3:
        pass
    elif datasource == DataSource.LOCAL:
        # Recursively iterate over all the files and subdirectories in the current directory
        for root, dirs, files in os.walk(source):
            for file in files:
                path = os.path.join(root, file)
                try:
                    data = Image.open(path)
                except FileNotFoundError:
                    print("The supplied file does not exist {}".format(path))
                    continue
                except UnidentifiedImageError:
                    print("The supplied file is not an image {}".format(path))
                    continue
                except Exception as e:
                    print("Error while reading file {}".format(path))
                    print(e)
                    continue

                encodings_json = llm_model.get_media_encoding(data)
                vector_database.add([encodings_json.get("embedding")], [path], [path])
    else:
        raise ValueError("Invalid data source")


def query(query: str, llm_model: BaseLLM, vector_database: BaseVectorDatabase) -> None:
    encodings_json = llm_model.get_text_encoding(query)
    vector_database.query(encodings_json.get("embedding"), 1)


def _infer_type(source: str) -> DataSource:
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
