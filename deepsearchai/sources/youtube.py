import json
import os
import uuid
from pathlib import Path

import googleapiclient.discovery
import pytube
import requests

try:
    from langchain.document_loaders import (
        GoogleApiClient,
        GoogleApiYoutubeLoader,
        YoutubeLoader,
    )
except ImportError:
    raise ImportError(
        'YouTube video requires extra dependencies. Install with `pip install --upgrade "deepsearch[dataloaders]"`'
    ) from None

from ..embedding_models.base import BaseEmbeddingModel
from ..embedding_models_config import EmbeddingModelsConfig
from ..enums import MEDIA_TYPE
from ..vector_databases.base import BaseVectorDatabase
from .base import BaseSource
from .data_source import DataSource


class YoutubeDatasource(BaseSource):
    OUTPUT_PATH = "tmp/deepsearch/youtube/"

    def __init__(self):
        # Create a YouTube API service object
        self.youtube_client = googleapiclient.discovery.build(
            "youtube", "v3", developerKey=os.environ.get("GOOGLE_CLIENT_API_KEY")
        )
        super().__init__()

    def add_data(
        self,
        source: str,
        embedding_models_config: EmbeddingModelsConfig,
        vector_database: BaseVectorDatabase,
    ) -> None:
        channel_id = source.split(":")[1]
        video_ids = self._get_channel_video_ids(channel_id)
        for video_id in video_ids:
            data = self._chunk_and_load_video(video_id)
            embedding_models = embedding_models_config.get_embedding_model(
                MEDIA_TYPE.VIDEO
            )
            for embedding_model in embedding_models:
                vector_database.add(
                    data,
                    DataSource.LOCAL,
                    video_id,
                    source,
                    MEDIA_TYPE.VIDEO,
                    embedding_model,
                )

    def _chunk_and_load_video(self, video_id):
        # Download the audio of the video
        yt = pytube.YouTube(f"https://www.youtube.com/watch?v={video_id}")
        audio = yt.streams.filter(only_audio=True).first()
        filename = "{}/{}".format(self.OUTPUT_PATH, audio.default_filename)
        audio.download(output_path=self.OUTPUT_PATH)
        return filename

    def _get_channel_video_ids(self, channel_id):
        """Gets the video IDs for a YouTube channel.

        Args:
            channel_id: The ID of the YouTube channel.

        Returns:
            A list of video IDs for the YouTube channel.
        """
        channel_resource = (
            self.youtube_client.channels()
            .list(id=channel_id, part="contentDetails")
            .execute()
        )

        # Get the channel's upload playlist ID
        upload_playlist_id = channel_resource["items"][0]["contentDetails"][
            "relatedPlaylists"
        ]["uploads"]

        # Retrieve the playlist's video list
        playlist_items = (
            self.youtube_client.playlistItems()
            .list(part="snippet", playlistId=upload_playlist_id)
            .execute()
        )

        # Get the video IDs from the playlist's video items
        video_ids = []
        for item in playlist_items["items"]:
            video_ids.append(item["snippet"]["resourceId"]["videoId"])

        return video_ids
