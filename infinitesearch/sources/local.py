from .base import BaseSource


class LocalDataSource(BaseSource):
    def __init__(sel):
        super().__init__()

    def add_data(self, source: str):
        return open(source).read()