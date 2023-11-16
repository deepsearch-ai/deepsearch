import unittest
from unittest import mock
from unittest.mock import patch

import mock.mock

from deepsearchai.enums import MEDIA_TYPE
from deepsearchai.sources.data_source import DataSource
from deepsearchai.sources.local import LocalDataSource


class LocalDataSourceTest(unittest.TestCase):
    def setUp(self):
        self.local_data_source = LocalDataSource()

    @patch("os.walk")
    @patch("PIL.Image.open")
    def test_add_data_image_directory_with_no_existing_files(
        self, mock_image_file, mock_listdir
    ):
        embedding_models_config = mock.Mock()
        embedding_model = mock.Mock()

        embedding_models_config.get_embedding_model.return_value = [embedding_model]

        # Create a mock image file
        image_data = mock.Mock()
        mock_listdir.return_value = [
            ("test_directory", "", ["image1.jpg", "image2.png"])
        ]
        mock_image_file.return_value = image_data

        # Create a mock vector database
        vector_database = mock.Mock()
        vector_database.get_existing_document_ids.return_value = []

        # Add local datasource data for a local directory
        directory = "test_directory"
        self.local_data_source.add_data(
            directory, embedding_models_config, vector_database
        )
        assert vector_database.add.mock_calls == [
            mock.call(
                image_data,
                DataSource.LOCAL,
                "test_directory/image1.jpg",
                directory,
                MEDIA_TYPE.IMAGE,
                embedding_model,
            ),
            mock.call(
                image_data,
                DataSource.LOCAL,
                "test_directory/image2.png",
                directory,
                MEDIA_TYPE.IMAGE,
                embedding_model,
            ),
        ]

    @patch("os.walk")
    @patch("PIL.Image.open")
    def test_add_data_image_directory_with_no_existing_files(
        self, mock_image_file, mock_listdir
    ):
        embedding_models_config = mock.Mock()
        embedding_model = mock.Mock()

        embedding_models_config.get_embedding_model.return_value = [embedding_model]

        # Create a mock image file
        image_data = mock.Mock()
        mock_listdir.return_value = [
            ("test_directory", "", ["image1.jpg", "image2.png"])
        ]
        mock_image_file.return_value = image_data

        # Create a mock vector database
        vector_database = mock.Mock()
        vector_database.get_existing_document_ids.return_value = []

        # Add local datasource data for a local directory
        directory = "test_directory"
        self.local_data_source.add_data(
            directory, embedding_models_config, vector_database
        )
        assert vector_database.add.mock_calls == [
            mock.call(
                image_data,
                DataSource.LOCAL,
                "test_directory/image1.jpg",
                directory,
                MEDIA_TYPE.IMAGE,
                embedding_model,
            ),
            mock.call(
                image_data,
                DataSource.LOCAL,
                "test_directory/image2.png",
                directory,
                MEDIA_TYPE.IMAGE,
                embedding_model,
            ),
        ]

    @patch("os.walk")
    @patch("PIL.Image.open")
    def test_add_data_image_directory_with_existing_files(
        self, mock_image_file, mock_listdir
    ):
        embedding_models_config = mock.Mock()
        embedding_model = mock.Mock()

        embedding_models_config.get_embedding_model.return_value = [embedding_model]

        # Create a mock image file
        image_data = mock.Mock()
        mock_listdir.return_value = [
            ("test_directory", "", ["image1.jpg", "image2.png"])
        ]
        mock_image_file.return_value = image_data

        # Create a mock vector database, such that one file already exists in it
        vector_database = mock.Mock()
        vector_database.get_existing_document_ids.return_value = [
            "test_directory/image2.png"
        ]

        # Add local datasource data for a local directory
        directory = "test_directory"
        self.local_data_source.add_data(
            directory, embedding_models_config, vector_database
        )
        assert vector_database.add.mock_calls == [
            mock.call(
                image_data,
                DataSource.LOCAL,
                "test_directory/image1.jpg",
                directory,
                MEDIA_TYPE.IMAGE,
                embedding_model,
            )
        ]

    @patch("os.path.isfile")
    @patch("PIL.Image.open")
    @patch("mimetypes.guess_type")
    def test_add_data_image(self, mock_mimetype, mock_image_file, mock_isfile):
        embedding_models_config = mock.Mock()
        embedding_model = mock.Mock()

        embedding_models_config.get_embedding_model.return_value = [embedding_model]

        image_data = mock.Mock()
        mock_isfile.return_value = True
        mock_image_file.return_value = image_data
        mock_mimetype.return_value = "image/jpeg", "encoding"

        # Create a mock vector database, such that one file already exists in it
        vector_database = mock.Mock()
        vector_database.get_existing_document_ids.return_value = []
        directory = "image1.jpg"

        # Add local datasource data for a local directory
        self.local_data_source.add_data(
            directory, embedding_models_config, vector_database
        )
        assert vector_database.add.mock_calls == [
            mock.call(
                image_data,
                DataSource.LOCAL,
                directory,
                directory,
                MEDIA_TYPE.IMAGE,
                embedding_model,
            )
        ]

    @patch("os.path.isfile")
    @patch("mimetypes.guess_type")
    def test_add_data_audio(self, mock_mimetypes, mock_is_file):
        embedding_models_config = mock.Mock()
        embedding_model = mock.Mock()

        embedding_models_config.get_embedding_model.return_value = [embedding_model]
        # Create a mock image file
        mock_is_file.return_value = True
        mock_mimetypes.return_value = "audio/mp3", "encoding"

        # Create a mock vector database, such that one file already exists in it
        vector_database = mock.Mock()
        vector_database.get_existing_document_ids.return_value = []

        filename = "audio.mp3"
        # Add local datasource data for a local directory
        self.local_data_source.add_data(
            filename, embedding_models_config, vector_database
        )
        assert vector_database.add.mock_calls == [
            mock.call(
                filename,
                DataSource.LOCAL,
                filename,
                filename,
                MEDIA_TYPE.AUDIO,
                embedding_model,
            )
        ]

    @patch("os.path.isfile")
    @patch("mimetypes.guess_type")
    def test_add_data_unsupported_media(self, mock_mimetype, mock_isfile):
        embedding_models_config = mock.Mock()
        embedding_model = mock.Mock()

        embedding_models_config.get_embedding_model.return_value = [embedding_model]
        # Create a mock unsupported file
        # Create a mock image file
        mock_isfile.return_value = True
        mock_mimetype.return_value = "random/format", "encoding"

        # Create a mock vector database, such that one file already exists in it
        vector_database = mock.Mock()
        vector_database.get_existing_document_ids.return_value = []

        # Create a mock for the llm model
        filename = "random.format"
        # Add local datasource data for a local directory
        self.local_data_source.add_data(
            filename, embedding_models_config, vector_database
        )
        vector_database.add.assert_not_called()
