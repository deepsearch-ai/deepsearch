from typing import Dict, List, Optional

from typing_extensions import TypedDict

from deepsearchai.enums import MEDIA_TYPE


class MediaData(TypedDict):
    document: str
    metadata: Optional[Dict[str, str]]


class QueryResult(TypedDict):
    llm_response: str
    documents: Optional[Dict[MEDIA_TYPE, List[MediaData]]]
