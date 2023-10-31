from .base import BaseSource
from ..llms.base import BaseLLM
from ..vector_databases.base import BaseVectorDatabase

import os
import boto3
import base64
import io
from infinitesearch.llms.clip import Clip
from PIL import Image, UnidentifiedImageError

import urllib.parse


class S3DataSource(BaseSource):
    def __init__(self):
        self.access_key = os.environ.get("AWS_ACCESS_KEY")
        self.secret_key = os.environ.get("AWS_SECRET_KEY")
        self.client = boto3.client('s3',
                                   aws_access_key_id=self.access_key,
                                   aws_secret_access_key=self.secret_key,
                                   region_name="us-east-1")
        super().__init__()

    # add a particular file
    def add_data(self, source: str, llm_model: BaseLLM, vector_database: BaseVectorDatabase) -> None:
        bucket_name = self._get_s3_bucket_name(source)
        key = self._get_s3_object_key_name(source)
        objects = self._get_all_objects_inside_an_object(bucket_name, key)
        for s3_object in objects:
            data = self._load_image_from_s3(bucket_name, s3_object)
            encoded_image = llm_model.get_media_encoding(data)
            vector_database.add(encoded_image.get("embedding"), ["{}.{}".format(bucket_name, s3_object)],
                                ["{}.{}".format(bucket_name, s3_object)])

    def _load_image_from_s3(self, bucket_name, object_key):
        """Loads an image from S3 and opens it using PIL.

        Args:
          client: A boto3 S3 client object.
          bucket_name: The name of the S3 bucket.
          object_key: The key of the image object.

        Returns:
          A PIL Image object.
        """

        response = self.client.get_object(Bucket=bucket_name, Key=object_key)
        image_data = response["Body"].read()

        image_stream = io.BytesIO(image_data)
        return Image.open(image_stream)

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
        """Gets all the objects inside an object in S3.

        Args:
          client: A boto3 S3 client object.
          bucket_name: The name of the S3 bucket.
          object_key: The key of the object to list its contents.

        Returns:
          A list of the names of all the objects inside the object.
        """

        objects = []
        if not object_key:
            response = self.client.list_objects_v2(Bucket=bucket_name, Prefix='')
        else:
            response = self.client.list_objects_v2(Bucket=bucket_name, Prefix=object_key + "/")
        while True:
            for s3_object in response["Contents"]:
                objects.append(s3_object["Key"])

            if "NextContinuationToken" in response:
                response = self.client.list_objects_v2(Bucket=bucket_name, Prefix=object_key + "/",
                                                       ContinuationToken=response["NextContinuationToken"])
            else:
                break
        return objects
