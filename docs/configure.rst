Configure our RAG App
===============

This step is OPTIONAL as we have defaults for all configuration values.
But, you might need API_KEYs for some default options -  OpenAI and Youtube.

Embedding Models
===============

You can choose one embedding model that you prefer for each data type.



STEP 1: To use DeepSearch you need to add data

Instructions to add local data from the CLI
Open Python Shell and run the following commands

.. code-block:: console

    from deepsearch.app import App
    from deepsearch.vector_databases.chromadb import ChromaDB
    from deepsearch.llms.clip import Clip

    llm_model = Clip()
    db = ChromaDB()
    app = App(None, llm_model, db)

    app.add_data(<LOCAL_PATH>)
