import os

from PIL import Image, UnidentifiedImageError

from deepsearchai.embedding_models_config import EmbeddingModelsConfig
from deepsearchai.enums import MEDIA_TYPE
from deepsearchai.utils import get_mime_type
from deepsearchai.vector_databases.base import BaseVectorDatabase
from .base import BaseSource
from .data_source import DataSource


class LocalDataSource(BaseSource):
    def __init__(self):
        super().__init__()

    def add_data(
        self,
        source: str,
        embedding_models_config: EmbeddingModelsConfig,
        vector_database: BaseVectorDatabase,
    ) -> None:
        # Recursively iterate over all the files and subdirectories in the current directory
        existing_document_identifiers = {}
        file_paths = self._get_all_file_path(source)
        for file in file_paths:
            media_type = get_mime_type(file)
            embedding_models = embedding_models_config.get_embedding_model(media_type)
            for embedding_model in embedding_models:
                if media_type not in existing_document_identifiers:
                    existing_document_identifiers[
                        media_type
                    ] = vector_database.get_existing_document_ids(
                        {"document_id": file_paths},
                        embedding_model.get_collection_name(media_type),
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
                vector_database.add(
                    data, DataSource.LOCAL, file, source, media_type, embedding_model
                )

    def _get_all_file_path(self, directory):
        if os.path.isfile(directory):
            return [directory]

        file_paths = []
        for root, _, files in os.walk(directory):
            for file in files:
                file_path = os.path.join(root, file)
                file_paths.append(file_path)
        return file_paths
