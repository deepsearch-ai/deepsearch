import unittest
from unittest import mock
from unittest.mock import patch

from deepsearch.enums import MEDIA_TYPE
from deepsearch.vector_databases.chromadb import ChromaDB
from deepsearch.vector_databases.configs.chromadb import ChromaDbConfig


class ChromaDBTest(unittest.TestCase):
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
                    name=config.audio_collection_name,
                    embedding_function=config.embedding_function,
                    metadata={"hnsw:space": "cosine"},
                ),
                mock.call(
                    name=config.image_collection_name,
                    embedding_function=config.embedding_function,
                    metadata={"hnsw:space": "cosine"},
                ),
                mock.call(
                    name=config.video_collection_name,
                    embedding_function=config.embedding_function,
                    metadata={"hnsw:space": "cosine"},
                ),
            ],
        )

    @patch("chromadb.Client")
    def test_add(self, chromadb_client):
        config = ChromaDbConfig()
        chromadb = ChromaDB(config=config)

        mock_image_collection = chromadb_client.return_value.get_or_create_collection(
            name=config.image_collection_name,
            embedding_function=config.embedding_function,
            metadata={"hnsw:space": "cosine"},
        )

        embeddings = [[1.0, 2.0], [3.0, 4.0]]
        documents = ["doc1", "doc2"]
        ids = ["id1", "id2"]
        metadatas = [{"key1": "value1"}, {"key2": "value2"}]
        data_type = MEDIA_TYPE.IMAGE

        chromadb.add(embeddings, documents, ids, metadatas, data_type)

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
        mock_image_collection = mock.Mock()
        mock_audio_collection = mock.Mock()
        mock_video_collection = mock.Mock()
        chromadb_client.return_value.get_or_create_collection.side_effect = [
            mock_audio_collection,
            mock_image_collection,
            mock_video_collection,
        ]

        config = ChromaDbConfig()
        chromadb = ChromaDB(config=config)

        mock_image_collection.query.return_value = {
            "ids": [["document_image_id1", "document_image_id2"]],
            "documents": [
                [
                    "This is image document 1",
                    "This is image document 2",
                ],
            ],
            "metadatas": [[{"source": "imagesource1"}, {"source": "imagesource2"}]],
            "distances": [[0.321456789, 0.456789012]],
        }
        mock_audio_collection.query.return_value = {
            "ids": [["document_audio_id1", "document_audio_id2"]],
            "embeddings": [[[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]]],
            "documents": [
                [
                    "This is audio document 1",
                    "This is audio document 2",
                ],
            ],
            "metadatas": [[{"source": "audiosource1"}, {"source": "audiosource2"}]],
            "distances": [[0.36, 0.56]],
        }
        input_query = "This is a query"
        input_embeddings = [1.0, 2.0, 3.0]
        n_results = 10

        results = chromadb.query(
            input_query, input_embeddings, n_results, MEDIA_TYPE.IMAGE, 0.5
        )
        self.assertEqual(
            results,
            [
                {
                    "document": "This is image document 1",
                    "metadata": {"source": "imagesource1"},
                },
                {
                    "document": "This is image document 2",
                    "metadata": {"source": "imagesource2"},
                },
            ],
        )

        results = chromadb.query(
            input_query, input_embeddings, n_results, MEDIA_TYPE.AUDIO, 0.5
        )
        self.assertEqual(
            results,
            [
                {
                    "document": "This is audio document 1",
                    "metadata": {"source": "audiosource1"},
                }
            ],
        )

    @patch("chromadb.Client")
    def test_get_existing_document_ids(self, chromadb_client):
        mock_image_collection = mock.Mock()
        mock_audio_collection = mock.Mock()
        mock_video_collection = mock.Mock()
        chromadb_client.return_value.get_or_create_collection.side_effect = [
            mock_audio_collection,
            mock_image_collection,
            mock_video_collection,
        ]

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

        config = ChromaDbConfig()
        chromadb = ChromaDB(config=config)
        metadata_filters = {"key1": "value1"}
        data_type = MEDIA_TYPE.IMAGE

        self.assertEqual(
            chromadb.get_existing_document_ids(metadata_filters, data_type),
            ["document1"],
        )
        mock_audio_collection.get.assert_not_called()
        mock_video_collection.get.assert_not_called()

    @patch("chromadb.Client")
    def test_count(self, chromadb_client):
        mock_image_collection = mock.Mock()
        mock_audio_collection = mock.Mock()
        mock_video_collection = mock.Mock()
        chromadb_client.return_value.get_or_create_collection.side_effect = [
            mock_audio_collection,
            mock_image_collection,
            mock_video_collection,
        ]

        config = ChromaDbConfig()
        chromadb = ChromaDB(config=config)

        mock_image_collection.count.return_value = 100
        mock_audio_collection.count.return_value = 200
        mock_video_collection.count.return_value = 300

        count = chromadb.count()

        assert count == {
            "image_collection": 100,
            "audio_collection": 200,
            "video_collection": 300,
        }

    @patch("chromadb.Client")
    def test_delete(self, chromadb_client):
        mock_image_collection = mock.Mock()
        mock_audio_collection = mock.Mock()
        mock_video_collection = mock.Mock()
        chromadb_client.return_value.get_or_create_collection.side_effect = [
            mock_audio_collection,
            mock_image_collection,
            mock_video_collection,
        ]

        config = ChromaDbConfig()
        chromadb = ChromaDB(config=config)

        chromadb.delete({})

        mock_image_collection.delete.assert_called_once()
        mock_audio_collection.delete.assert_called_once()
        mock_video_collection.delete.assert_called_once()

        mock_image_collection.reset_mock()
        mock_audio_collection.reset_mock()
        mock_video_collection.reset_mock()

        chromadb.delete({}, media_type=MEDIA_TYPE.IMAGE)

        mock_image_collection.delete.assert_called_once()
        mock_audio_collection.delete.assert_not_called()
        mock_video_collection.delete.assert_not_called()

        mock_image_collection.reset_mock()
        mock_audio_collection.reset_mock()
        mock_video_collection.reset_mock()

        chromadb.delete({}, media_type=MEDIA_TYPE.AUDIO)

        mock_audio_collection.delete.assert_called_once()
        mock_image_collection.delete.assert_not_called()
        mock_video_collection.delete.assert_not_called()

        mock_image_collection.reset_mock()
        mock_audio_collection.reset_mock()
        mock_video_collection.reset_mock()

        chromadb.delete({}, media_type=MEDIA_TYPE.VIDEO)

        mock_video_collection.delete.assert_called_once()
        mock_image_collection.delete.assert_not_called()
        mock_audio_collection.delete.assert_not_called()
