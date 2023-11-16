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


We use CLIP(https://github.com/openai/CLIP) as our embedding model for images, but can be extended to support other embedding models as well.
Similarly, we use Whisper(https://github.com/openai/whisper) as our embedding model for audio, but can be extended to support other embedding models as well.