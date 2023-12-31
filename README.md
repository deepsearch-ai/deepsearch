<p align="center">
<img src="docs/images/logo_cropped.png" alt="DeepSearch Logo">
</p>

## 🔧 Quick install

```bash
pip install --upgrade deepsearchai
```

## 🔍 Usage and Demo

<!-- Demo GIF or Image -->
<p align="center">
  <img src="docs/images/demo.gif" width="900px" alt="Deepsearch Demo">
</p>


A python lib to search for relevant media from your data corpus. This library not only helps you search the relevant media but also highlights the most relevant section of the media. For example, given a list of audio files you can search over them to find the exact second of the file that is relevant to your search query.

1. Users can upload images, text, audios or videos to add as data to be searched
2. LLM models (customisable) help tag, describe and index the data
3. Embeddings from this are then added to a vector database (configurable)
4. Search APIs that help relevant data (across all multimedia types) from the vector database.

Our documentation [here](https://deepsearch.readthedocs.io/en/latest/) has a more updated set of instructions and details

# Instructions to run the UI

1. Navigate to the root folder of the repo
2. poetry shell && poetry lock && poetry install
3. streamlit run deepsearchai/streamlit_app.py

Alternately, you can run
```
from deepsearch.app import App
app = App()

app.run()
```

# Instructions to add local data from the CLI

1. Open Python Shell and run the following commands

```
from deepsearch.app import App
app = App()

app.add_data(<LOCAL_PATH>)
```

# Instructions to add S3 data from the CLI

1. Open Python Shell and run the following commands

```
from deepsearch.app import App
app = App()

app.add_data(<S3_PATH>)
```

# Audio Usage

1. Having ffmpeg installed is a prerequisite. All major package installers include ffmpeg,
   but if you need to manually install it, see [here](https://www.ffmpeg.org/download.html). You may
   refer https://github.com/openai/whisper/blob/main/README.md

You can only have 1 model per datatype. For example: Clip for images and Whisper for Audio.
Only one vector_database. Embedding from each datatype will go into a new collection.

The collection names would default to audio_collection, image_collection but can be overriden with a database config file.

# Video Usage

1. We currently support loading a youtube channel
2. A youtube channel ID can be obtained from https://www.youtube.com/account_advanced
3. For a channel you dont own, the channel ID is the last part of the URL. For example, the channel ID for https://www.youtube.com/channel/UCXuqSBlHAE6Xw-yeJA0Tunw is UCXuqSBlHAE6Xw-yeJA0Tunw

# Setting up Keys for Video

Set 'GOOGLE_CLIENT_API_KEY' environment variable.

# Pytorch issues
1. Install pytorch via pip causes issues for Mac M1/M2 machines. For such cases, you might have to install pytorch via conda. Please refer https://www.geeksforgeeks.org/how-to-install-pytorch-on-macos/

# Triton Issues
1. Installing triton can be tricky with Mac M1. This happens when attempting to install the audio dependency for the package. Please remote triton manually from the poetry.lock file if the issue persists.

## Citation
If you utilize this repository, please consider citing it with:

```
@misc{deepsearch,
  author = {Rupesh Bansal, Shiwangi Shah},
  title = {Deepsearch: Semantic search on multimedia sources like audio, video and images},
  year = {2023},
  publisher = {GitHub},
  journal = {GitHub repository},
  howpublished = {\url{https://github.com/deepsearch-ai/deepsearch}},
}
```

# License

This project is licensed under the Apache License, Version 2.0. Please see the LICENSE: LICENSE file for more information.