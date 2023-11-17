Getting started
----------------------------
There are two simple ways to get started with the application.

1. Using our UI built using streamlit

2. Python CLI

To install deepsearchai, run
.. code-block:: console

    pip install deepsearchai

Run the UI using Streamlit
==============================
1. By default, deepsearchai is not shipped with dependencies to render the UI elements. If you wish to use the UI, run
.. code-block:: console

    pip install deepsearchai[ui]


2. Create an app instance and run it

.. code-block:: console

    from deepsearchai.app import App
    app = App()
    app.run()

.. figure:: /images/upload_ui.png
   :alt: UI screenshot 1
   :align: center

   Fig: Add Data to the App via UI

.. figure:: /images/query_ui.png
   :alt: UI screenshot 2
   :align: center

   Fig: Query the App via UI

.. figure:: /images/response_ui.png
   :alt: UI screenshot 3
   :align: center

   Fig: Response generated

Running the App from the CLI
==============================

Initialising the App
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Open terminal, create appropriate virtual environment and install the package

.. code-block:: console

    pip install deepsearchai

By default, the package is shipped with dependencies to process images. If you also want to process audios, run

.. code-block:: console

        pip install deepsearchai[audio]


If you wish to process videos, run

.. code-block:: console

        pip install deepsearchai[video]


Open Python Shell and run the following commands

.. code-block:: console

    from deepsearchai.app import App

    app = App()

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

Youtube Channel
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
Deepsearch lets users index youtube channels. This is a great way to index videos from a channel, and query them later. The indexing happens in a way such that queries will surface the exact second from where the relevant context as per the user query is present

.. code-block:: console

    app.add_data("youtube:<YOUTUBE_CHANNEL_ID>")

Deepsearch automatically infers the datasource from the path provided. If the path is a local path, it will be treated as a local file. If the path is a youtube channel, it will be treated as a youtube channel. If the path is an S3 path, it will be treated as an S3 path.

Querying
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: console

    from deepsearchai.enums import MEDIA_TYPE
    app.query(<input string query>, [List of media_types to search across])

For example

.. code-block:: console

    from deepsearchai.enums import MEDIA_TYPE
    app.query("A car in front of a building", [MEDIA_TYPE.IMAGE, MEDIA_TYPE.AUDIO])


Users can add 3 types of data to the deepsearch

#. Images : This can be images in any standard format (jpg, png, jpeg, etc). Deepsearch will infer the type, and process it accordingly.
#. Audios : This can be audios in any standard format (mp3, wav, etc). Deepsearch will infer the type, and process it accordingly.
#. Videos: This has to be a youtube video channel. Deepsearch has validations which will prevent users from attempting to inject videos in any other format.