import unittest
from unittest import mock
from unittest.mock import patch

from deepsearchai.embedding_models.blip_image_captioning import \
    BlipImageCaptioning
from deepsearchai.embedding_models.clip import Clip
from deepsearchai.embedding_models.whisper_openai import WhisperOpenAi
from deepsearchai.enums import MEDIA_TYPE
from deepsearchai.sources.data_source import DataSource
from deepsearchai.vector_databases.chromadb import ChromaDB
from deepsearchai.vector_databases.configs.chromadb import ChromaDbConfig


class ChromaDBTest(unittest.TestCase):
    blip_image_captioning = BlipImageCaptioning()
    whisper_openai = WhisperOpenAi()
    clip = Clip()

    @patch("chromadb.Client")
    def test_init(self, chromadb_client):
        config = ChromaDbConfig()
        chromadb = ChromaDB(config=config)

        self.assertEqual(chromadb.config, config)
        self.assertEqual(chromadb.client, chromadb_client.return_value)
        self.assertEqual(
            chromadb_client.return_value.get_or_create_collection.mock_calls,
            [
                mock.call(
                    name=self.whisper_openai.get_collection_name(MEDIA_TYPE.AUDIO),
                    embedding_function=config.embedding_function,
                    metadata={"hnsw:space": "cosine"},
                ),
                mock.call(
                    name=self.clip.get_collection_name(MEDIA_TYPE.IMAGE),
                    embedding_function=config.embedding_function,
                    metadata={"hnsw:space": "cosine"},
                ),
                mock.call(
                    name=self.blip_image_captioning.get_collection_name(
                        MEDIA_TYPE.IMAGE
                    ),
                    embedding_function=config.embedding_function,
                    metadata={"hnsw:space": "cosine"},
                ),
                mock.call(
                    name=self.whisper_openai.get_collection_name(MEDIA_TYPE.VIDEO),
                    embedding_function=config.embedding_function,
                    metadata={"hnsw:space": "cosine"},
                ),
            ],
        )

    @patch("chromadb.Client")
    def test_add(self, chromadb_client):
        # Mock clip model to be able to mock the corresponding generated embeddings
        clip_model_mock = mock.Mock()
        clip_model_mock.get_collection_name.return_value = (
            self.clip.get_collection_name(MEDIA_TYPE.IMAGE)
        )

        # Mock embedding models config, and make it return the mocked clip model
        embedding_models_config = mock.Mock()
        embedding_models_config.llm_models.items.return_value = [
            (MEDIA_TYPE.IMAGE, [clip_model_mock])
        ]

        # Initialize chromadb with the mocks
        config = ChromaDbConfig()
        chromadb = ChromaDB(
            embedding_models_config=embedding_models_config, config=config
        )

        mock_image_collection = chromadb_client.return_value.get_or_create_collection(
            name=self.clip.get_collection_name(MEDIA_TYPE.IMAGE),
            embedding_function=config.embedding_function,
            metadata={"hnsw:space": "cosine"},
        )

        embeddings = [[1.0, 2.0]]
        documents = ["file"]
        ids = ["id1"]
        metadatas = [
            {"source_type": "LOCAL", "source_id": "source", "document_id": "file"}
        ]

        clip_model_mock.get_media_encoding.return_value = {
            "embedding": embeddings,
            "ids": ids,
        }

        chromadb.add(
            "local_file_path",
            DataSource.LOCAL,
            "file",
            "source",
            MEDIA_TYPE.IMAGE,
            clip_model_mock,
        )

        mock_image_collection.add.assert_called_once()
        self.assertEqual(
            mock_image_collection.add.mock_calls,
            [
                mock.call(
                    embeddings=embeddings,
                    documents=documents,
                    ids=ids,
                    metadatas=metadatas,
                )
            ],
        )

    @patch("chromadb.Client")
    def test_query(self, chromadb_client):
        # Mock clip model to be able to mock the corresponding generated embeddings
        clip_model_mock = mock.Mock()
        clip_model_mock.get_collection_name.return_value = (
            self.clip.get_collection_name(MEDIA_TYPE.IMAGE)
        )

        # Mock embedding models config, and make it return the mocked clip model
        embedding_models_config = mock.Mock()
        embedding_models_config.llm_models.items.return_value = [
            (MEDIA_TYPE.IMAGE, [clip_model_mock])
        ]

        # Initialize chromadb with the mocks
        config = ChromaDbConfig()
        chromadb = ChromaDB(
            embedding_models_config=embedding_models_config, config=config
        )

        mock_image_collection = chromadb_client.return_value.get_or_create_collection(
            name=self.clip.get_collection_name(MEDIA_TYPE.IMAGE),
            embedding_function=config.embedding_function,
            metadata={"hnsw:space": "cosine"},
        )

        mock_image_collection.query.return_value = {
            "ids": [["document_image_id1", "document_image_id2"]],
            "documents": [
                [
                    "This is image document 1",
                    "This is image document 2",
                ],
            ],
            "metadatas": [[{"source": "imagesource1"}, {"source": "imagesource2"}]],
            "distances": [[0.721456789, 0.456789012]],
        }

        input_query = "This is a query"
        input_embeddings = [1.0, 2.0, 3.0]
        n_results = 10

        clip_model_mock.get_text_encodding.return_value = {
            "embedding": input_embeddings,
        }

        results = chromadb.query(
            input_query, n_results, MEDIA_TYPE.IMAGE, 0.5, clip_model_mock
        )
        self.assertEqual(
            results,
            [
                {
                    "document": "This is image document 1",
                    "metadata": {"source": "imagesource1"},
                }
            ],
        )

    @patch("chromadb.Client")
    def test_get_existing_document_ids(self, chromadb_client):
        # Mock clip model to be able to mock the corresponding generated embeddings
        collection_name = self.clip.get_collection_name(MEDIA_TYPE.IMAGE)
        clip_model_mock = mock.Mock()
        clip_model_mock.get_collection_name.return_value = collection_name

        # Mock embedding models config, and make it return the mocked clip model
        embedding_models_config = mock.Mock()
        embedding_models_config.llm_models.items.return_value = [
            (MEDIA_TYPE.IMAGE, [clip_model_mock])
        ]

        # Initialize chromadb with the mocks
        config = ChromaDbConfig()
        chromadb = ChromaDB(
            embedding_models_config=embedding_models_config, config=config
        )

        mock_image_collection = chromadb_client.return_value.get_or_create_collection(
            name=self.clip.get_collection_name(MEDIA_TYPE.IMAGE),
            embedding_function=config.embedding_function,
            metadata={"hnsw:space": "cosine"},
        )

        mock_image_collection.get.side_effect = [
            {
                "ids": ["id1"],
                "metadatas": [{"document_id": "document1"}],
            },
            {
                "ids": [],
                "metadatas": [],
            },
        ]

        metadata_filters = {"key1": "value1"}
        self.assertEqual(
            chromadb.get_existing_document_ids(metadata_filters, collection_name),
            ["document1"],
        )

    @patch("chromadb.Client")
    def test_count(self, chromadb_client):
        mock_image_collection = mock.Mock()
        mock_image_caption_collection = mock.Mock()
        mock_audio_collection = mock.Mock()
        mock_video_collection = mock.Mock()
        chromadb_client.return_value.get_or_create_collection.side_effect = [
            mock_audio_collection,
            mock_image_collection,
            mock_image_caption_collection,
            mock_video_collection,
        ]

        config = ChromaDbConfig()
        chromadb = ChromaDB(config=config)

        mock_image_collection.count.return_value = 100
        mock_image_caption_collection.count.return_value = 150
        mock_audio_collection.count.return_value = 200
        mock_video_collection.count.return_value = 300

        count = chromadb.count()

        assert count == {
            mock_image_collection.name: 100,
            mock_image_caption_collection.name: 150,
            mock_audio_collection.name: 200,
            mock_video_collection.name: 300,
        }

    @patch("chromadb.Client")
    def test_delete(self, chromadb_client):
        mock_image_collection = mock.Mock()
        mock_image_caption_collection = mock.Mock()
        mock_audio_collection = mock.Mock()
        mock_video_collection = mock.Mock()
        chromadb_client.return_value.get_or_create_collection.side_effect = [
            mock_audio_collection,
            mock_image_collection,
            mock_image_caption_collection,
            mock_video_collection,
        ]

        config = ChromaDbConfig()
        chromadb = ChromaDB(config=config)

        chromadb.delete({})

        mock_image_collection.delete.assert_called_once()
        mock_image_caption_collection.delete.assert_called_once()
        mock_audio_collection.delete.assert_called_once()
        mock_video_collection.delete.assert_called_once()

        mock_image_collection.reset_mock()
        mock_image_caption_collection.reset_mock()
        mock_audio_collection.reset_mock()
        mock_video_collection.reset_mock()

        chromadb.delete({}, media_type=MEDIA_TYPE.IMAGE)

        mock_image_collection.delete.assert_called_once()
        mock_image_caption_collection.delete.assert_called_once()
        mock_audio_collection.delete.assert_not_called()
        mock_video_collection.delete.assert_not_called()

        mock_image_collection.reset_mock()
        mock_image_caption_collection.reset_mock()
        mock_audio_collection.reset_mock()
        mock_video_collection.reset_mock()

        chromadb.delete({}, media_type=MEDIA_TYPE.AUDIO)

        mock_audio_collection.delete.assert_called_once()
        mock_image_collection.delete.assert_not_called()
        mock_image_caption_collection.delete.assert_not_called()
        mock_video_collection.delete.assert_not_called()

        mock_image_collection.reset_mock()
        mock_image_caption_collection.reset_mock()
        mock_audio_collection.reset_mock()
        mock_video_collection.reset_mock()

        chromadb.delete({}, media_type=MEDIA_TYPE.VIDEO)

        mock_video_collection.delete.assert_called_once()
        mock_image_collection.delete.assert_not_called()
        mock_image_caption_collection.delete.assert_not_called()
        mock_audio_collection.delete.assert_not_called()
