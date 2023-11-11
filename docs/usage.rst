Usage
=====

Installation
------------

.. code-block:: console

    git clone


Usage
------------

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
