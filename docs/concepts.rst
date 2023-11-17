Fundamentals
======================================

We are a multimodal RAG Application that generates embeddings based on the datatype. We also auto generate captions for the images for improved accuracy. All parts of the model are configurable.
The sequence diagrams below explain the various steps of computation:


.. figure:: /images/add_data_seq.png
   :alt: sequence diagram
   :align: center

   Fig: Add Data to the App

.. figure:: /images/query.png
   :alt: sequence diagram 2
   :align: center

   Fig: Query the App

Deepsearch exposes 2 main functionalities, both via UI and CLI to the end user

#. Add Data: This lets users add data to the app. We currently support data from 3 main datasources
    * S3: This can be an S3 bucket or its sub folder. For this to work, users need to configure the environment variables AWS_ACCESS_KEY, AWS_SECRET_KEY with read object permissions on the buckets of interest. Deepsearch has inherent optimizations to not index an already indexed object.
    * Local: This can be local file or a folder path. Deepsearch will scan through the supplied path, and index all the supported files mimetypes. Deepsearch has inherent optimizations to not index an already indexed object.
    * Youtube Channel: This has to be a youtube channel in the format <youtube:CHANNEL_ID>. Users need to set YOUTUBE_API_KEY environment variable to be able use this faeture. Steps to generate this token can be found in https://developers.google.com/youtube/registering_an_application. Deepsearch smartly chunks and stores all the videos such that queries will return the exact second of the video where the match is found. This is a very powerful feature for video search. Like other datasources, we do not index an already indexed video.
#. Query Data: This lets users query the app to search for relevant media. Users can explicitly provide a list of MEDIA_TYPE to search for. Deepsearch supports 3 main media types
    * Image: This is a simple image query. Deepsearch will return the top k results based on the query. The results are sorted based on the cosine similarity of the query and the embedding of the data.
    * Audio: This is a simple audio query. Deepsearch will return the top k results based on the query, with the precise second of the audio where the match occured. The results are sorted based on the cosine similarity of the query and the embedding of the data.
    * Video: This is a simple video query. Deepsearch will return the top k results based on the query, with the precise second of the audio where the match occured. The results are sorted based on the cosine similarity of the query and the embedding of the data.

We use CLIP(https://github.com/openai/CLIP) as our embedding model for images, but can be extended to support other embedding models as well.
Similarly, we use Whisper(https://github.com/openai/whisper) as our embedding model for audio, but can be extended to support other embedding models as well.