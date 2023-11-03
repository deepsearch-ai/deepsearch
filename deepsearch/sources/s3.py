import hashlib
import io
import os
import urllib.parse

from ..enums import MEDIA_TYPE
import boto3
from PIL import Image, UnidentifiedImageError

from ..llms_config import LlmsConfig
from ..vector_databases.base import BaseVectorDatabase
from .base import BaseSource
from ..utils import get_mime_type
from ..enums import MEDIA_TYPE


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
            self, source: str, llms_config: LlmsConfig, vector_database: BaseVectorDatabase
    ) -> None:
        # test with a subdirectory
        bucket_name = self._get_s3_bucket_name(source)
        key = self._get_s3_object_key_name(source)
        objects = self._get_all_objects_inside_an_object(bucket_name, key)
        object_identifiers = self._get_object_identifiers(bucket_name, objects)
        existing_object_identifiers = vector_database.get_existing_object_identifiers(
            object_identifiers
        )
        for s3_object, identifier in zip(objects, object_identifiers):
            if identifier in existing_object_identifiers:
                "{} already exists, skipping...".format(
                    "s3://{}/{}".format(bucket_name, s3_object)
                )
                continue
            media_type = get_mime_type(s3_object)
            if media_type == MEDIA_TYPE.IMAGE:
                media_data = self._load_image_from_s3(bucket_name, s3_object)
                if media_data is None:
                    continue
            elif media_type == MEDIA_TYPE.AUDIO:
                media_data = self._load_audio_from_s3(bucket_name, s3_object)
            else:
                print("Unsupported media type {}".format(s3_object))
                continue
            data = llms_config.get_llm_model(media_type).get_media_encoding(media_data)
            # We should ideally batch upload the data to the vector database.
            vector_database.add(
                [data.get("embedding")],
                ["s3://{}/{}".format(bucket_name, s3_object)],
                [identifier]
            )

    def _load_audio_from_s3(self, bucket_name, s3_object):
        pass

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
        if not object_key:
            response = self.client.list_objects_v2(Bucket=bucket_name)
        else:
            response = self.client.list_objects_v2(
                Bucket=bucket_name, Prefix=object_key + "/"
            )

        while True:
            for object in response["Contents"]:
                if object["Key"].endswith("/"):
                    continue
                files.append(object["Key"])

            if "NextContinuationToken" in response:
                response = self.client.list_objects_v2(
                    Bucket=bucket_name,
                    Prefix=object_key + "/",
                    ContinuationToken=response["NextContinuationToken"],
                )
            else:
                break
        return files

    def _get_object_identifiers(self, bucket_name, objects):
        ids = []
        for s3_object in objects:
            ids.append(
                hashlib.sha256(
                    ("{}/{}".format(bucket_name, s3_object)).encode()
                ).hexdigest()
            )

        return ids
