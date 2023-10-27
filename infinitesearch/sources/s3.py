from .base import BaseSource


class S3DataSource(BaseSource):
    def __init__(self):
        super().__init__()

    def add_data(self, source: str):
        return open(source).read()
