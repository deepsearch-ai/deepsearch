import unittest
from unittest import mock
from unittest.mock import patch

import mock.mock

from ..enums import MEDIA_TYPE
from ..llms_config import LlmsConfig
from ..utils import get_mime_type
from ..vector_databases.base import BaseVectorDatabase
from ..vector_databases.configs.base import BaseVectorDatabaseConfig
from ..sources.base import BaseSource
from ..sources.local import LocalDataSource


class LocalDataSourceTest(unittest.TestCase):

    def setUp(self):
        self.llms_config = LlmsConfig()
        self.vector_database = BaseVectorDatabase(BaseVectorDatabaseConfig())
        self.local_data_source = LocalDataSource()

    @patch("os.listdir")
    @patch("PIL.Image.open")
    def test_add_data_image_directory(self, mock_image_file, mock_listdir):
        # Create a mock image file
        mock_listdir.return_value = ['image1.jpg', 'image2.png']
        mock_image_file.return_value = mock.Mock()

        vector_database = mock.Mock()
        vector_database.get_existing_document_ids.return_value = []

        embeddings = [[0, 0, 0]]
        metadata = [{"author": "author1"}]
        ids = ["id1"]
        encodings_json = {
            "embedding": embeddings,
            "metadata": metadata,
            "ids": ids
        }
        llms_config = mock.Mock()
        llms_config.get_llm_model.return_value.get_media_encoding.return_value = encodings_json
        directory = 'test_directory'
        self.local_data_source.add_data(directory, llms_config, vector_database)
        vector_database.add.assert_called_with(embeddings, [], ids, metadata, MEDIA_TYPE.IMAGE)

    def test_add_data_image(self):
        # Create a mock image file
        image_file = mock.patch('PIL.Image.open', return_value=mock.Mock())
        mock.patch('os.path.isfile', return_value=True)
        with mock.patch('PIL.Image.open', image_file):
            file_path = 'test_image.jpg'
            self.local_data_source.add_data('test', self.llms_config, self.vector_database)
            self.vector_database.add.assert_called_once()

    def test_add_data_audio(self):
        # Create a mock audio file
        audio_file = mock.patch('os.path.join', return_value='test_audio.mp3')
        with mock.patch('os.path.join', audio_file):
            file_path = 'test_audio.mp3'
            self.local_data_source.add_data('test', self.llms_config, self.vector_database)
            self.vector_database.add.assert_called_once()

    def test_add_data_unsupported_media(self):
        # Create a mock unsupported file
        unsupported_file = mock.patch('os.path.join', return_value='test_unsupported.txt')
        with mock.patch('os.path.join', unsupported_file):
            file_path = 'test_unsupported.txt'
            self.local_data_source.add_data('test', self.llms_config, self.vector_database)
            self.vector_database.add.assert_not_called()

    def test_get_all_file_path(self):
        # Create a mock directory
        mock_directory = mock.patch('os.listdir', return_value=['file1.jpg', 'file2.mp3', 'file3.txt'])
        with mock.patch('os.listdir', mock_directory):
            directory = 'test_directory'
            full_paths = self.local_data_source._get_all_file_path(directory)
            self.assertEqual(full_paths,
                             ['test_directory/file1.jpg', 'test_directory/file2.mp3', 'test_directory/file3.txt'])
