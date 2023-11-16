Introduction to Deepsearch
======================================
A python lib to search for the relevant media from your data corpus. This library not only helps you search the relevant media but also *highlights the most relevant section of the media*. For example, given a list of audio files you can search over them to find the subsection of the file that is relevant to your search query.

Features the library supports:

1. Users can upload images, text, audios or videos to add as data to be searched

2. LLM models (customisable) help tag, describe and index the data

3. Embeddings from this are then added to a vector database (configurable)

4. Accept search queries and provide relevant data (across all multimedia types) from the vector database. Also, provide an answer from an LLM using the relevant data matches.

Deepsearch currently supports chromaDB and its default vector database, but can be extended to support other vector databases as well.
