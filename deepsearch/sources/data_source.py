from enum import Enum


# Supported datatypes
class DataSource(Enum):
    LOCAL = 1
    S3 = 2
    YOUTUBE = 3
