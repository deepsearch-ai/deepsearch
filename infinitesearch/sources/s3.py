from .base import BaseSource
import os
import boto3

class S3DataSource(BaseSource):
    def __init__(self):
        super().__init__()

    def add_data(self, source: str):

        accessKey = os.environ.get("AWS_ACCESS_KEY")
        secretKey = os.environ.get("AWS_SECRET_KEY")
        s3 = boto3.client('s3', accessKey , secretKey)

        s3.download_file('BUCKET_NAME', 'OBJECT_NAME', 'FILE_NAME')
        return open(source).read()
