import unittest

from infinitesearch.sources.s3 import S3DataSource


class TestS3(unittest.TestCase):

    def test_add_data(self):
        s3_data_source = S3DataSource()

        # Test adding a file from S3
        s3_data_source.add_data("s3", "ai-infinitesearch", "_1cc5049d-d07b-46c8-897b-bf2aebffbc6e.jpeg")

        # Verify that the file was added to the llm model
        self.assertTrue(Clip.has_data("downloads/tmp"))
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
