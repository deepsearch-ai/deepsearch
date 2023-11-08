import unittest
from unittest import mock
from unittest.mock import patch

import mock.mock

from deepsearch.enums import MEDIA_TYPE
from deepsearch.sources.local import LocalDataSource


class LocalDataSourceTest(unittest.TestCase):
    def setUp(self):
        self.local_data_source = LocalDataSource()

    @patch("os.listdir")
    @patch("PIL.Image.open")
    def test_add_data_image_directory_with_no_existing_files(
        self, mock_image_file, mock_listdir
    ):
        # Create a mock image file
        mock_listdir.return_value = ["image1.jpg", "image2.png"]
        mock_image_file.return_value = mock.Mock()

        # Create a mock vector database
        vector_database = mock.Mock()
        vector_database.get_existing_document_ids.return_value = []

        # Create a mock for the arguments to be received from the llm model
        embeddings = [[0, 0, 0]]
        metadata = [{"author": "author1"}]
        ids = ["id1"]
        encodings_json = {"embedding": embeddings, "metadata": metadata, "ids": ids}

        # Create a mock for the llm model
        llms_config = mock.Mock()
        llms_config.get_llm_model.return_value.get_media_encoding.return_value = (
            encodings_json
        )

        # Add local datasource data for a local directory
        directory = "test_directory"
        self.local_data_source.add_data(directory, llms_config, vector_database)
        assert vector_database.add.mock_calls == [
            mock.call(
                embeddings,
                ["test_directory/image1.jpg"],
                ids,
                metadata,
                MEDIA_TYPE.IMAGE,
            ),
            mock.call(
                embeddings,
                ["test_directory/image2.png"],
                ids,
                metadata,
                MEDIA_TYPE.IMAGE,
            ),
        ]

    @patch("os.listdir")
    @patch("PIL.Image.open")
    def test_add_data_image_directory_with_no_existing_files(
        self, mock_image_file, mock_listdir
    ):
        # Create a mock image file
        mock_listdir.return_value = ["image1.jpg", "image2.png"]
        mock_image_file.return_value = mock.Mock()

        # Create a mock vector database
        vector_database = mock.Mock()
        vector_database.get_existing_document_ids.return_value = []

        # Create a mock for the arguments to be received from the llm model
        embeddings = [[0, 0, 0]]
        metadata = [{"author": "author1"}]
        ids = ["id1"]
        encodings_json = {"embedding": embeddings, "metadata": metadata, "ids": ids}

        # Create a mock for the llm model
        llms_config = mock.Mock()
        llms_config.get_llm_model.return_value.get_media_encoding.return_value = (
            encodings_json
        )

        new_metadata_1 = [
            {
                "source_type": "LOCAL",
                "source_id": "test_directory",
                "document_id": "test_directory/image1.jpg",
                "author": "author1",
            }
        ]
        new_metadata_2 = [
            {
                "source_type": "LOCAL",
                "source_id": "test_directory",
                "document_id": "test_directory/image2.png",
                "author": "author1",
            }
        ]
        # Add local datasource data for a local directory
        directory = "test_directory"
        self.local_data_source.add_data(directory, llms_config, vector_database)
        assert vector_database.add.mock_calls == [
            mock.call(
                embeddings,
                ["test_directory/image1.jpg"],
                ids,
                new_metadata_1,
                MEDIA_TYPE.IMAGE,
            ),
            mock.call(
                embeddings,
                ["test_directory/image2.png"],
                ids,
                new_metadata_2,
                MEDIA_TYPE.IMAGE,
            ),
        ]

    @patch("os.listdir")
    @patch("PIL.Image.open")
    def test_add_data_image_directory_with_existing_files(
        self, mock_image_file, mock_listdir
    ):
        # Create a mock image file
        mock_listdir.return_value = ["image1.jpg", "image2.png"]
        mock_image_file.return_value = mock.Mock()

        # Create a mock vector database, such that one file already exists in it
        vector_database = mock.Mock()
        vector_database.get_existing_document_ids.return_value = [
            "test_directory/image2.png"
        ]

        # Create a mock for the arguments to be received from the llm model
        embeddings = [[0, 0, 0]]
        metadata = [{"author": "author1"}]
        ids = ["id1"]
        encodings_json = {"embedding": embeddings, "metadata": metadata, "ids": ids}

        new_metadata = [
            {
                "source_type": "LOCAL",
                "source_id": "test_directory",
                "document_id": "test_directory/image1.jpg",
                "author": "author1",
            }
        ]
        # Create a mock for the llm model
        llms_config = mock.Mock()
        llms_config.get_llm_model.return_value.get_media_encoding.return_value = (
            encodings_json
        )

        # Add local datasource data for a local directory
        directory = "test_directory"
        self.local_data_source.add_data(directory, llms_config, vector_database)
        assert vector_database.add.mock_calls == [
            mock.call(
                embeddings,
                ["test_directory/image1.jpg"],
                ids,
                new_metadata,
                MEDIA_TYPE.IMAGE,
            )
        ]

    @patch("os.path.isfile")
    @patch("PIL.Image.open")
    @patch("mimetypes.guess_type")
    def test_add_data_image(self, mock_mimetype, mock_image_file, mock_isfile):
        mock_isfile.return_value = True
        mock_image_file.return_value = mock.Mock()
        mock_mimetype.return_value = "image/jpeg", "encoding"

        # Create a mock vector database, such that one file already exists in it
        vector_database = mock.Mock()
        vector_database.get_existing_document_ids.return_value = []

        # Create a mock for the arguments to be received from the llm model
        embeddings = [[0, 0, 0]]
        metadata = [{"author": "author1"}]
        ids = ["id1"]
        encodings_json = {"embedding": embeddings, "metadata": metadata, "ids": ids}

        directory = "image1.jpg"
        # Create a mock for the llm model
        llms_config = mock.Mock()
        llms_config.get_llm_model.return_value.get_media_encoding.return_value = (
            encodings_json
        )
        new_metadata = [
            {
                "source_type": "LOCAL",
                "source_id": directory,
                "document_id": directory,
                "author": "author1",
            }
        ]
        # Add local datasource data for a local directory
        self.local_data_source.add_data(directory, llms_config, vector_database)
        assert vector_database.add.mock_calls == [
            mock.call(embeddings, [directory], ids, new_metadata, MEDIA_TYPE.IMAGE)
        ]

    @patch("os.path.isfile")
    @patch("mimetypes.guess_type")
    def test_add_data_audio(self, mock_mimetypes, mock_is_file):
        # Create a mock image file
        mock_is_file.return_value = True
        mock_mimetypes.return_value = "audio/mp3", "encoding"

        # Create a mock vector database, such that one file already exists in it
        vector_database = mock.Mock()
        vector_database.get_existing_document_ids.return_value = []

        # Create a mock for the arguments to be received from the llm model
        embeddings = [[0, 0, 0]]
        metadata = [{"author": "author1"}]
        ids = ["id1"]
        encodings_json = {"embedding": embeddings, "metadata": metadata, "ids": ids}

        # Create a mock for the llm model
        llms_config = mock.Mock()
        llms_config.get_llm_model.return_value.get_media_encoding.return_value = (
            encodings_json
        )
        filename = "audio.mp3"
        new_metadata = [
            {
                "source_type": "LOCAL",
                "source_id": filename,
                "document_id": filename,
                "author": "author1",
            }
        ]
        # Add local datasource data for a local directory
        self.local_data_source.add_data(filename, llms_config, vector_database)
        assert vector_database.add.mock_calls == [
            mock.call(embeddings, [filename], ids, new_metadata, MEDIA_TYPE.AUDIO)
        ]

    @patch("os.path.isfile")
    @patch("mimetypes.guess_type")
    def test_add_data_unsupported_media(self, mock_mimetype, mock_isfile):
        # Create a mock unsupported file
        # Create a mock image file
        mock_isfile.return_value = True
        mock_mimetype.return_value = "random/format", "encoding"

        # Create a mock vector database, such that one file already exists in it
        vector_database = mock.Mock()
        vector_database.get_existing_document_ids.return_value = []

        # Create a mock for the llm model
        llms_config = mock.Mock()
        filename = "random.format"
        # Add local datasource data for a local directory
        self.local_data_source.add_data(filename, llms_config, vector_database)
        vector_database.add.assert_not_called
        llms_config.get_llm_model.assert_not_called
