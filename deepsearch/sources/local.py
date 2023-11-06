import os

from PIL import Image, UnidentifiedImageError

from ..enums import MEDIA_TYPE
from ..llms_config import LlmsConfig
from ..utils import get_mime_type
from ..vector_databases.base import BaseVectorDatabase
from .base import BaseSource


class LocalDataSource(BaseSource):
    def __init__(self):
        super().__init__()

    def add_data(
            self, source: str, llms_config: LlmsConfig, vector_database: BaseVectorDatabase
    ) -> None:
        # Recursively iterate over all the files and subdirectories in the current directory
        existing_document_identifiers = {}
        file_paths = self._get_all_file_path(source)
        for file in file_paths:
            media_type = get_mime_type(file)
            if media_type not in existing_document_identifiers:
                existing_document_identifiers[media_type] = vector_database.get_existing_document_ids(
                    {"document_id": file_paths}, media_type
                )

            if file in existing_document_identifiers[media_type]:
                "{} already exists, skipping...".format(file)
                continue
            if media_type == MEDIA_TYPE.IMAGE:
                try:
                    data = Image.open(file)
                except FileNotFoundError:
                    print("The supplied file does not exist {}".format(file))
                    continue
                except UnidentifiedImageError:
                    print("The supplied file is not an image {}".format(file))
                    continue
                except Exception as e:
                    print("Error while reading file {}".format(file))
                    print(e)
                    continue

            elif media_type == MEDIA_TYPE.AUDIO:
                data = file
            else:
                print("Unsupported media type {}".format(file))
                continue
            encodings_json = llms_config.get_llm_model(media_type).get_media_encoding(
                data, media_type
            )
            import pdb; pdb.set_trace()
            embeddings = encodings_json.get("embedding", None)
            documents = [file] if not encodings_json.get("documents") else encodings_json.get("documents")
            metadata = self._construct_metadata(
                encodings_json.get("metadata", None), source, file, len(documents)
            )
            ids = encodings_json.get("ids", [])
            vector_database.add(embeddings, documents, ids, metadata, media_type)

    def _get_all_file_path(self, directory):
        if os.path.isfile(directory):
            return [directory]
        # Get all the files in the supplied path folder
        files = os.listdir(directory)

        # Get the full absolute path of each file
        full_paths = []
        for file in files:
            full_path = os.path.join(directory, file)
            full_paths.append(full_path)
        return full_paths
