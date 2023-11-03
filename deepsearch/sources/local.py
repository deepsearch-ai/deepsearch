import os

from PIL import Image, UnidentifiedImageError

from ..enums import MEDIA_TYPE
from ..llms.base import BaseLLM
from ..vector_databases.base import BaseVectorDatabase
from .base import BaseSource


class LocalDataSource(BaseSource):
    def __init__(self):
        super().__init__()

    def add_data(
        self, source: str, llm_model: BaseLLM, vector_database: BaseVectorDatabase, data_type: MEDIA_TYPE
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

                encodings_json = llm_model.get_media_encoding(data, data_type)

                embeddings = [encodings_json.get("embedding")]
                documents = [encodings_json.get("text") if not encodings_json.get("text") else path]
                metadata = []
                ids = [path]

                vector_database.add(embeddings, documents, ids)
