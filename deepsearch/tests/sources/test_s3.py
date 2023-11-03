import unittest

from deepsearch.llms.clip import Clip
from deepsearch.llms.whisper import Whisper
from deepsearch.llms_config import LlmsConfig
from deepsearch.sources.utils import SourceUtils
from deepsearch.vector_databases.chromadb import ChromaDB

from deepsearch.enums import MEDIA_TYPE


class TestS3(unittest.TestCase):
    # TODO: Handle full file path as input
    def test_add_data(self):
        utils = SourceUtils()
        ChromaDB().reset()
        # Test adding a file from S3
        utils.add_data("s3://ai-infinitesearch/test", LlmsConfig(), ChromaDB())

        matched_images = utils.query("A building", [MEDIA_TYPE.IMAGE], LlmsConfig(), ChromaDB())

        # Verify that the file was added to the llm model
        self.assertEqual(["s3://ai-infinitesearch/test/building.jpeg"], matched_images)

    def test_audio_add_data(self):
        utils = SourceUtils()
        ChromaDB().reset()
        # Test adding a file from S3
        utils.add_data(
            "s3://ai-infinitesearch/WhatsApp Ptt 2023-11-02 at 10.47.56.ogg",
            LlmsConfig(),
            ChromaDB(),
        )

        matched_files = utils.query("s3", [MEDIA_TYPE.AUDIO], LlmsConfig(), ChromaDB())

        # Verify that the file was added to the llm model
        self.assertEqual(
            ["s3://ai-infinitesearch/WhatsApp Ptt 2023-11-02 at 10.47.56.ogg"],
            matched_files,
        )

    def test_add_data_with_nested_folders(self):
        utils = SourceUtils()
        db = ChromaDB()
        db.reset()

        # Test adding a file from S3
        utils.add_data("s3://ai-infinitesearch/test/b", Clip(), db, MEDIA_TYPE.IMAGE)

        matched_images = utils.query("A building", Clip(), db)

        # Verify that the file was added to the llm model
        self.assertEqual(
            [
                "s3://ai-infinitesearch/test/b/_9ea8f598-fdee-45b7-9338-46bce1d2f3a4.jpeg"
            ],
            matched_images,
        )

    #
    # def test_add_data_with_invalid_source(self):
    #     s3_data_source = S3DataSource()
    #
    #     # Test adding a file from an invalid source
    #     with self.assertRaises(Exception):
    #         s3_data_source.add_data(source="invalid_source", bucket_name="my_bucket", object_name="my_object")
    #
    # def test_add_data_with_invalid_bucket_name(self):
    #     s3_data_source = S3DataSource()
    #
    #     # Test adding a file from an invalid bucket name
    #     with self.assertRaises(Exception):
    #         s3_data_source.add_data(source="s3", bucket_name="invalid_bucket", object_name="my_object")
    #
    # def test_add_data_with_invalid_object_name(self):
    #     s3_data_source = S3DataSource()
    #
    #     # Test adding a file from an invalid object name
    #     with self.assertRaises(Exception):
    #         s3_data_source.add_data(source="s3", bucket_name="my_bucket", object_name="invalid_object")
