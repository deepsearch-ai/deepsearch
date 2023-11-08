import io
import unittest
from unittest.mock import patch

import boto3
import mock
from PIL import Image, UnidentifiedImageError

from deepsearch.embedding_models import LlmsConfig
from deepsearch.enums import MEDIA_TYPE
from deepsearch.sources.base import BaseSource
from deepsearch.sources.s3 import S3DataSource
from deepsearch.utils import get_mime_type
from deepsearch.vector_databases.base import BaseVectorDatabase


class S3DataSourceTests(unittest.TestCase):
    @patch.object(boto3, "client")
    def setUp(self, mock_boto3_client):
        self.s3_data_source = S3DataSource()
        self.llms_config = LlmsConfig()
        self.mock_s3_client = mock_boto3_client.return_value

    def create_fake_image_data(self):
        image = Image.new("RGB", (100, 100), (255, 0, 0))
        image_bytes = io.BytesIO()
        image.save(image_bytes, format="JPEG")
        image_bytes.seek(0)
        return image_bytes

    def test_load_image_from_s3(self):
        bucket_name = "my-bucket"
        object_key = "my-image-file.jpg"
        self.mock_s3_client.get_object.return_value = {
            "Body": self.create_fake_image_data()
        }
        image = self.s3_data_source._load_image_from_s3(bucket_name, object_key)
        self.assertIsInstance(image, Image.Image)

    def test_get_all_objects_inside_an_object(self):
        bucket_name = "my-bucket"
        object_key = "my-folder"
        self.mock_s3_client.list_objects_v2.return_value = {
            "Contents": [
                {"Key": "my-folder/my-file1.jpg"},
                {"Key": "my-folder/my-file2.jpg"},
            ]
        }
        files, s3_paths = self.s3_data_source._get_all_objects_inside_an_object(
            bucket_name, object_key
        )
        self.assertEqual(files, ["my-folder/my-file1.jpg", "my-folder/my-file2.jpg"])
        self.assertEqual(
            s3_paths,
            [
                "s3://my-bucket/my-folder/my-file1.jpg",
                "s3://my-bucket/my-folder/my-file2.jpg",
            ],
        )

    def test_get_s3_bucket_name(self):
        url = "s3://my-bucket/my-folder/my-file.jpg"
        bucket_name = self.s3_data_source._get_s3_bucket_name(url)
        self.assertEqual(bucket_name, "my-bucket")

    def test_get_s3_object_key_name(self):
        url = "s3://my-bucket/my-folder/my-file.jpg"
        object_key = self.s3_data_source._get_s3_object_key_name(url)
        self.assertEqual(object_key, "my-folder/my-file.jpg")

    def test_add_data_for_image(self):
        source = "s3://my-bucket/my-folder/my-image.jpg"
        mock_vector_database = mock.Mock(BaseVectorDatabase)
        mock_vector_database.get_existing_document_ids.return_value = {}

        # Mock the get_all_objects_inside_an_object method to return a single object
        with patch.object(
            S3DataSource, "_get_all_objects_inside_an_object"
        ) as mock_get_all_objects_inside_an_object:
            mock_get_all_objects_inside_an_object.return_value = (
                ["my-image.jpg"],
                ["s3://my-bucket/my-folder/my-image.jpg"],
            )

            # Mock the load_image_from_s3 method to return a valid image
            with patch.object(
                S3DataSource, "_load_image_from_s3"
            ) as mock_load_image_from_s3:
                mock_load_image_from_s3.return_value = Image.new(
                    "RGB", (100, 100), (255, 0, 0)
                )

                # Create a mock for the arguments to be received from the llm model
                embeddings = [[0, 0, 0]]
                metadata = [{"author": "author1"}]
                ids = ["id1"]
                encodings_json = {
                    "embedding": embeddings,
                    "metadata": metadata,
                    "ids": ids,
                }

                directory = "s3://my-bucket/my-folder/my-image.jpg"
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

                self.s3_data_source.add_data(source, llms_config, mock_vector_database)
                mock_vector_database.add.assert_called_once()
                args, kwargs = mock_vector_database.add.call_args
                self.assertEqual(args[0], embeddings)
                self.assertEqual(args[1], [directory])
                self.assertEqual(args[2], ids)
                self.assertEqual(args[3], new_metadata)
                self.assertEqual(kwargs["data_type"], MEDIA_TYPE.IMAGE)

    def test_add_data_for_audio(self):
        source = "s3://my-bucket/my-folder/my-audio.mp3"
        bucket_name = "my-bucket"
        object_key = "my-folder/my-audio.mp3"
        mock_vector_database = mock.Mock(BaseVectorDatabase)
        mock_vector_database.get_existing_document_ids.return_value = {}

        # Mock the get_all_objects_inside_an_object method to return a single object
        with patch.object(
            S3DataSource, "_get_all_objects_inside_an_object"
        ) as mock_get_all_objects_inside_an_object:
            mock_get_all_objects_inside_an_object.return_value = (
                ["my-audio.mp3"],
                ["s3://my-bucket/my-folder/my-audio.mp3"],
            )

            # Mock the load_audio_from_s3 method to return a valid audio data
            with patch.object(
                S3DataSource, "_load_audio_from_s3"
            ) as mock_load_audio_from_s3:
                mock_load_audio_from_s3.return_value = io.BytesIO(b"fake audio data")

                # Create a mock for the arguments to be received from the llm model
                embeddings = [[0, 0, 0]]
                metadata = [{"author": "author1"}]
                ids = ["id1"]
                encodings_json = {
                    "embedding": embeddings,
                    "metadata": metadata,
                    "ids": ids,
                }

                directory = "s3://my-bucket/my-folder/my-audio.mp3"
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

                self.s3_data_source.add_data(source, llms_config, mock_vector_database)

        # Verify that the vector database add method was called with the correct arguments

        mock_vector_database.add.assert_called_once()
        args, kwargs = mock_vector_database.add.call_args

        self.assertEqual(args[0], embeddings)
        self.assertEqual(args[1], [directory])
        self.assertEqual(args[2], ids)
        self.assertEqual(args[3], new_metadata)
        self.assertEqual(kwargs["data_type"], MEDIA_TYPE.AUDIO)
