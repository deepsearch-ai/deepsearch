.. deepsearch documentation master file, created by
   sphinx-quickstart on Sat Nov 11 12:30:17 2023.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Documentation
======================================


.. toctree::
   :maxdepth: 2

   What is Deepsearch <intro>

   Getting started <getting_started>

   Concepts <concepts>

   Configuring the library <configure>

   Common Installation Issues <installation_issues>

Deepsearch is a python lib to search for relevant media from your data corpus. This library not only helps you search the relevant media but also highlights the most relevant section of the media. For example, given a list of audio files you can search over them to find the exact second of the file that is relevant to your search query.

1. Users can upload images, text, audios or videos to add as data to be searched
2. LLM models (customisable) help tag, describe and index the data
3. Embeddings from this are then added to a vector database (configurable)
4. Search APIs that help relevant data (across all multimedia types) from the vector database.

Deepsearch currently supports chromaDB and its default vector database, but can be extended to support other vector databases as well.
