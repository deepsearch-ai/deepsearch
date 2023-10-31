from .base import BaseSource
import os
import boto3
import base64
from infinitesearch.llms.clip import Clip
from PIL import Image, UnidentifiedImageError

class S3DataSource(BaseSource):
    def __init__(self):
        super().__init__()

    # add a particular file
    def add_data(self, source: str, bucket_name: str, object_name: str):
        accessKey = os.environ.get("AWS_ACCESS_KEY")
        secretKey = os.environ.get("AWS_SECRET_KEY")
        s3 = boto3.client('s3',
                          aws_access_key_id=accessKey,
                          aws_secret_access_key=secretKey,
                          region_name="us-east-1")

        with open('test123', 'wb') as f:
            s3.download_fileobj(bucket_name, object_name, f)
        f.close()
        clip = Clip()
        with open('test123', 'rb') as f:
            contents = base64.b64encode(f.read()).decode('utf-8')
        # add this file to llm model
        embeddings = clip.get_media_encoding(contents)
        f.close()
        return

    # add all files in a bucket
    # with a prefix can also be implemented
    def add_data_2(self, source: str, bucket_name: str, object_name: str, file_name: str):
        accessKey = os.environ.get("AWS_ACCESS_KEY")
        secretKey = os.environ.get("AWS_SECRET_KEY")
        s3 = boto3.client('s3', accessKey, secretKey)

        my_bucket = s3.Bucket(bucket_name)
        file_name = "tmp"
        output = f"downloads/{file_name}"
        for my_bucket_object in my_bucket.objects.all():
            s3.download_file(my_bucket_object, output)
            # add this file to llm model
            Clip.add_data(output, "s3", None)
        return
