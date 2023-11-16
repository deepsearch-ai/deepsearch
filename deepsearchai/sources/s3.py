import io
import os
import urllib.parse

import boto3
from PIL import Image, UnidentifiedImageError

from deepsearchai.embedding_models_config import EmbeddingModelsConfig
from deepsearchai.enums import MEDIA_TYPE
from deepsearchai.sources.base import BaseSource
from deepsearchai.sources.data_source import DataSource
from deepsearchai.utils import get_mime_type
from deepsearchai.vector_databases.base import BaseVectorDatabase


class S3DataSource(BaseSource):
    def __init__(self):
        self.access_key = os.environ.get("AWS_ACCESS_KEY")
        self.secret_key = os.environ.get("AWS_SECRET_KEY")
        self.client = boto3.client(
            "s3",
            aws_access_key_id=self.access_key,
            aws_secret_access_key=self.secret_key,
            region_name="us-east-1",
        )
        super().__init__()

    def add_data(
        self,
        source: str,
        embedding_models_config: EmbeddingModelsConfig,
        vector_database: BaseVectorDatabase,
    ) -> None:
        bucket_name = self._get_s3_bucket_name(source)
        key = self._get_s3_object_key_name(source)
        objects, s3_paths = self._get_all_objects_inside_an_object(bucket_name, key)
        existing_document_identifiers = {}
        for s3_object, object_s3_path in zip(objects, s3_paths):
            media_type = get_mime_type(s3_object)
            embedding_models = embedding_models_config.get_embedding_model(media_type)
            for embedding_model in embedding_models:
                if media_type not in existing_document_identifiers:
                    existing_document_identifiers[
                        media_type
                    ] = vector_database.get_existing_document_ids(
                        {"document_id": s3_paths},
                        embedding_model.get_collection_name(media_type),
                    )

                if object_s3_path in existing_document_identifiers[media_type]:
                    "{} already exists, skipping...".format(object_s3_path)
                    continue
                if media_type == MEDIA_TYPE.IMAGE:
                    data = self._load_image_from_s3(bucket_name, s3_object)
                    if data is None:
                        continue
                elif media_type == MEDIA_TYPE.AUDIO:
                    data = self._load_audio_from_s3(bucket_name, s3_object)
                else:
                    print("Unsupported media type {}".format(s3_object))
                    continue
                vector_database.add(
                    data,
                    DataSource.LOCAL,
                    object_s3_path,
                    source,
                    media_type,
                    embedding_model,
                )

    def _load_audio_from_s3(self, bucket_name, object_key):
        """Loads an audio file from S3 and returns the audio data."""
        # Create the directory if it doesn't exist
        if not os.path.exists("/tmp/deepsearch"):
            os.makedirs("/tmp/deepsearch")
        local_file_path = "/tmp/deepsearch/{}".format(object_key)
        self.client.download_file(bucket_name, object_key, local_file_path)
        return local_file_path

    def _load_image_from_s3(self, bucket_name, object_key):
        """Loads an image from S3 and opens it using PIL.

        Args:
          bucket_name: The name of the S3 bucket.
          object_key: The key of the image object.

        Returns:
          A PIL Image object.
        """

        response = self.client.get_object(Bucket=bucket_name, Key=object_key)
        image_data = response["Body"].read()

        image_stream = io.BytesIO(image_data)
        try:
            return Image.open(image_stream)
        except UnidentifiedImageError:
            print(
                "The supplied file is not an image {}".format(
                    "{}/{}".format(bucket_name, object_key)
                )
            )
            return None
        except Exception as e:
            print(
                "Error while reading file {}".format(
                    "{}/{}".format(bucket_name, object_key)
                )
            )
            print(e)
            return None

    def _get_s3_bucket_name(self, url):
        """Extracts the S3 bucket name from its URL.

        Args:
          url: The S3 URL.

        Returns:
          The S3 bucket name.
        """

        parsed_url = urllib.parse.urlparse(url, allow_fragments=False)
        return parsed_url.netloc

    def _get_s3_object_key_name(self, url):
        """Gets the S3 object name from its URL.

        Args:
          url: The S3 URL.

        Returns:
          The S3 object name.
        """

        parsed_url = urllib.parse.urlparse(url, allow_fragments=False)
        return parsed_url.path.strip("/")

    def _get_all_objects_inside_an_object(self, bucket_name, object_key):
        """Lists all the files inside a folder in an S3 bucket, but does not add the sub-folders.

        Args:
          bucket_name: The name of the S3 bucket.
          object_key: The key of the folder to list the files inside.

        Returns:
          A list of the names of all the files inside the folder.
        """

        files = []
        s3_paths = []
        if not object_key:
            response = self.client.list_objects_v2(Bucket=bucket_name)
        else:
            response = self.client.list_objects_v2(
                Bucket=bucket_name, Prefix=object_key
            )

        while True:
            for object in response["Contents"]:
                if object["Key"].endswith("/"):
                    continue
                files.append(object["Key"])
                s3_paths.append("s3://{}/{}".format(bucket_name, object["Key"]))

            if "NextContinuationToken" in response:
                response = self.client.list_objects_v2(
                    Bucket=bucket_name,
                    Prefix=object_key + "/",
                    ContinuationToken=response["NextContinuationToken"],
                )
            else:
                break
        return files, s3_paths
