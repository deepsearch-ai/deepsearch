import uuid

import requests
import json
from pathlib import Path
import googleapiclient.discovery
import os

try:
    from langchain.document_loaders import YoutubeLoader, GoogleApiYoutubeLoader, GoogleApiClient
except ImportError:
    raise ImportError(
        'YouTube video requires extra dependencies. Install with `pip install --upgrade "deepsearch[dataloaders]"`'
    ) from None

from .base import BaseSource
from ..vector_databases.base import BaseVectorDatabase
from ..embedding_models.base import BaseEmbeddingModel
from ..enums import MEDIA_TYPE


class YoutubeDatasource(BaseSource):
    VIDEO_CHUNK_LENGTH = 60000
    VIDEO_CHUNK_OVERLAP = 10000

    def __init__(self):
        self.google_api_client = GoogleApiClient(
            service_account_path=Path("~/.credentials/credentials.json")
        )
        # Create a YouTube API service object
        self.youtube_client = googleapiclient.discovery.build("youtube", "v3", developerKey=os.environ.get("GOOGLE_CLIENT_API_KEY"))
        super().__init__()

    def add_data(
            self,
            source: str,
            llm_model: BaseEmbeddingModel,
            vector_database: BaseVectorDatabase,
    ) -> None:
        video_ids = self._get_channel_video_ids(source)
        for video_id in video_ids:
            data = self._chunk_and_load_video(video_id)
            metadata = self._construct_metadata(data.get("metadata"), source, video_id, len(data.get("documents")))
            vector_database.add(None, data, data.get("ids"), metadata, MEDIA_TYPE.VIDEO)

    def _chunk_and_load_video(self, video_id):
        # Get the video's duration.
        video_duration = self.google_api_client.get_video_duration(video_id)

        start_time = 0
        end_time = start_time + self.VIDEO_CHUNK_LENGTH
        # For each chunk:
        #   Get the start time and end time of the chunk.
        #   Use the GoogleApiYoutubeLoader class to transcribe the chunk.
        #   Save the transcript of the chunk to a file.
        documents = []
        metadatas = []
        ids = []
        while end_time <= video_duration:
            chunk_transcript = self.google_api_client.load_transcript(video_id, start_time, end_time)
            documents.append(chunk_transcript)
            metadatas.append({
                "start": start_time,
                "end": end_time
            })
            ids.append(str(uuid.uuid4()))
            if end_time == video_duration:
                break
            start_time = start_time + self.VIDEO_CHUNK_LENGTH - self.VIDEO_CHUNK_OVERLAP
            end_time = min(start_time + self.VIDEO_CHUNK_LENGTH, video_duration)

    def _get_channel_video_ids(self, channel_id):
        """Gets the video IDs for a YouTube channel.

        Args:
            channel_id: The ID of the YouTube channel.

        Returns:
            A list of video IDs for the YouTube channel.
        """
        channel_resource = self.youtube_client.channels().list(part="contentDetails").execute()

        # Get the channel's upload playlist ID
        upload_playlist_id = channel_resource["items"][0]["contentDetails"]["relatedPlaylists"]["uploads"]

        # Retrieve the playlist's video list
        playlist_items = self.youtube_client.playlistItems().list(part="snippet", playlistId=upload_playlist_id).execute()

        # Get the video IDs from the playlist's video items
        video_ids = []
        for item in playlist_items["items"]:
            video_ids.append(item["snippet"]["resourceId"]["videoId"])

        return video_ids