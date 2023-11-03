import os

from PIL import Image, UnidentifiedImageError
from typing import Dict, Any
from ..enums import MEDIA_TYPE
from ..llms.base import BaseLLM
from ..llms_config import LlmsConfig
from ..vector_databases.base import BaseVectorDatabase
from .base import BaseSource
from ..enums import MEDIA_TYPE
from ..utils import get_mime_type
from .data_source import DataSource


class LocalDataSource(BaseSource):
    def __init__(self):
        super().__init__()

    def add_data(
            self, source: str, llms_config: LlmsConfig, vector_database: BaseVectorDatabase) -> None:
        # Recursively iterate over all the files and subdirectories in the current directory
        for root, dirs, files in os.walk(source):
            for file in files:
                path = os.path.join(root, file)
                media_type = get_mime_type(path)
                if media_type == MEDIA_TYPE.IMAGE:
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

                elif media_type == MEDIA_TYPE.AUDIO:
                    data = path
                else:
                    print("Unsupported media type {}".format(path))
                    continue
                encodings_json = llms_config.get_llm_model(media_type).get_media_encoding(data, media_type)
                embeddings = encodings_json.get("embedding", None)
                documents = [path] if not encodings_json.get("documents") else encodings_json.get("documents")
                metadata = self._construct_metadata(encodings_json.get("metadata", None), source, path, len(documents))
                ids = encodings_json.get("ids", [])
                vector_database.add(embeddings, documents, ids, metadata, media_type)
