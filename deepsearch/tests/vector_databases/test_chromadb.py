import unittest
from unittest.mock import patch

from ....deepsearch.vector_databases.chromadb import ChromaDB
from ....deepsearch.vector_databases.configs.chromadb import ChromaDbConfig

class ChromaDBTest(unittest.TestCase):

    @patch("chromadb.Client")
    def test_init(self, chromadb_client):
        config = ChromaDbConfig()
        chromadb = ChromaDB(config=config)
        assert chromadb.config == config
        assert chromadb.client == chromadb_client.return_value
        assert chromadb.image_collection == chromadb_client.return_value.get_or_create_collection.call_args[1]['name']
        assert chromadb.audio_collection == chromadb_client.return_value.get_or_create_collection.call_args[0]['name']

    def test_add(chromadb_client):
        config = ChromaDbConfig()
        chromadb = ChromaDB(config=config)
        embeddings = [[1.0, 2.0], [3.0, 4.0]]
        documents = ['doc1', 'doc2']
        ids = ['id1', 'id2']
        metadatas = [{'key1': 'value1'}, {'key2': 'value2'}]
        data_type = MEDIA_TYPE.IMAGE

        chromadb.add(embeddings, documents, ids, metadatas, data_type)

        chromadb_client.return_value.image_collection.add.assert_called_once()
        assert chromadb_client.return_value.image_collection.add.call_args[0]['embeddings'] == embeddings
        assert chromadb_client.return_value.image_collection.add.call_args[1]['documents'] == documents
        assert chromadb_client.return_value.image_collection.add.call_args[2]['ids'] == ids
        assert chromadb_client.return_value.image_collection.add.call_args[3]['metadatas'] == metadatas

    def test_query(chromadb_client):
        config = ChromaDbConfig()
        chromadb = ChromaDB(config=config)
        input_query = 'This is a query'
        input_embeddings = [1.0, 2.0, 3.0]
        n_results = 10
        data_types = [MEDIA_TYPE.IMAGE, MEDIA_TYPE.AUDIO]

        results = chromadb.query(input_query, input_embeddings, n_results, data_types)

        chromadb_client.return_value.image_collection.query.assert_called_once()
        chromadb_client.return_value.audio_collection.query.assert_called_once()
        assert results == ['result1', 'result2']

    def test_get_existing_document_ids(chromadb_client):
        config = ChromaDbConfig()
        chromadb = ChromaDB(config=config)
        metadata_filters = {'key1': 'value1'}
        data_type = MEDIA_TYPE.IMAGE

        results = chromadb.get_existing_document_ids(metadata_filters, data_type)

        chromadb_client.return_value.image_collection.get.assert_called_once()
        assert results == ['doc1', 'doc2']

    def test_get_collection(chromadb_client):
        config = ChromaDbConfig()
        chromadb = ChromaDB(config=config)

        collection = chromadb.get_collection()

        assert collection == chromadb_client.return_value.image_collection

    def test_count(chromadb_client):
        config = ChromaDbConfig()
        chromadb = ChromaDB(config=config)

        chromadb_client.return_value.image_collection.count.return_value = 100
        chromadb_client.return_value.audio_collection.count.return_value = 200

        count = chromadb.count()

        assert count == {'image_collection': 100, 'audio_collection': 200}

    def test_delete(chromadb_client):
        config = ChromaDbConfig
