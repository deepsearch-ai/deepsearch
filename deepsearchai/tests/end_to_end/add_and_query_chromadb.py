import unittest

from deepsearchai.app import App
from deepsearchai.embedding_models.clip import Clip
from deepsearchai.embedding_models_config import EmbeddingModelsConfig
from deepsearchai.enums import MEDIA_TYPE
from deepsearchai.llms.openai import OpenAi
from deepsearchai.sources.utils import SourceUtils
from deepsearchai.vector_databases.chromadb import ChromaDB


class TestS3(unittest.TestCase):
    # TODO: Handle full file path as input
    def test_add_s3_and_get_data_image(self):
        ChromaDB().reset()
        app = App(None, None, None)
        # Test adding a file from S3
        app.add_data("s3://ai-infinitesearch")

        matched_images = app.get_data("A monument", [MEDIA_TYPE.IMAGE])

        # Verify that the file was added to the llm model
        self.assertEqual(["s3://ai-infinitesearch/test/building.jpeg"], matched_images)

    def test_audio_add_data(self):
        app = App(None, None, None)
        ChromaDB().reset()
        # Test adding a file from S3
        app.add_data(
            "s3://ai-infinitesearch/WhatsApp Ptt 2023-11-02 at 10.47.56.ogg",
            EmbeddingModelsConfig(),
            ChromaDB(),
        )

        matched_files = app.query(
            "s3", [MEDIA_TYPE.AUDIO], EmbeddingModelsConfig(), ChromaDB()
        )

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
