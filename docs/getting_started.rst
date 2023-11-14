Getting started
----------------------------
There are two simple ways to get started with the application.

1. Using our UI hosted built using streamlit

2. Python CLI

Run our UI using Streamlit
==============================
1. Navigate to the root folder of the repo
2. Install all dependencies

.. code-block:: console

    poetry shell && poetry lock && poetry install
3. Run the webapp

.. code-block:: console

    streamlit run streamlit_deepsearch/streamlit_app.py

Running the App from the CLI
==============================

Initialising the App
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Open Python Shell and run the following commands

.. code-block:: console

    from deepsearch.app import App
    from deepsearch.vector_databases.chromadb import ChromaDB
    from deepsearch.llms.clip import Clip

    llm_model = Clip()
    db = ChromaDB()
    app = App(None, llm_model, db)

Add Data
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Local Files
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: console

    app.add_data(<LOCAL_PATH>)

S3 data from the CLI
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: console

    app.add_data(<S3_PATH>)


Querying
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: console

    app.query(<input string query>, [List of media_types to search across])

For example

.. code-block:: console

    app.query("A car in front of a building", [MEDIA_TYPE.IMAGE, MEDIA_TYPE.AUDIO])