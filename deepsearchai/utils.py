import mimetypes

from deepsearchai.enums import MEDIA_TYPE


def get_mime_type(filename: str) -> MEDIA_TYPE:
    mime_type, encoding = mimetypes.guess_type(filename)
    if not mime_type or mime_type.split("/")[0].upper() not in MEDIA_TYPE.__members__:
        return MEDIA_TYPE.UNKNOWN
    return MEDIA_TYPE[mime_type.split("/")[0].upper()]
