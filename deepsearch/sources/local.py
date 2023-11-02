import os

from PIL import Image, UnidentifiedImageError

from ..llms.base import BaseLLM
from ..vector_databases.base import BaseVectorDatabase
from .base import BaseSource


class LocalDataSource(BaseSource):
    def __init__(self):
        super().__init__()

    def add_data(
        self, source: str, llm_model: BaseLLM, vector_database: BaseVectorDatabase
    ) -> None:
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
