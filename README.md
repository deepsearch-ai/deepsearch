# deepsearch
A simple python lib to search for relevant media from your data corpus.
1. Users can upload images, text, audios or videos to add as data to be searched
2. LLM models (customisable) help tag, describe and index the data
3. Embeddings from this are then added to a vector database (configurable)
4. Search APIs that help relevant data (across all multimedia types) from the vector database.


# Instructions to run the UI
1. Navigate to the root folder of the repo
2. poetry shell && poetry lock && poetry install
3. streamlit run streamlit_deepsearch/streamlit_app.py


# Instructions to add local data from the CLI
1. Open Python Shell and run the following commands

```
from deepsearch.app import App
from deepsearch.vector_databases.chromadb import ChromaDB
from deepsearch.llms.clip import Clip

llm_model = Clip()
db = ChromaDB()
app = App(None, llm_model, db)

app.add_data(<LOCAL_PATH>)
```

# Instructions to add S3 data from the CLI
1. Open Python Shell and run the following commands

```
from deepsearch.app import App
from deepsearch.vector_databases.chromadb import ChromaDB
from deepsearch.llms.clip import Clip

llm_model = Clip()
db = ChromaDB()
app = App(None, llm_model, db)

app.add_data(<S3_PATH>)
```