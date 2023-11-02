import logging

import streamlit as st

from deepsearch.app import App
from deepsearch.llms.clip import Clip
from deepsearch.vector_databases.chromadb import ChromaDB

app = App(None, Clip(), ChromaDB())


def local_css(file_name):
    with open(file_name) as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)


def remote_css(url):
    st.markdown(f'<link href="{url}" rel="stylesheet">', unsafe_allow_html=True)


def icon(icon_name):
    st.markdown(f'<i class="material-icons">{icon_name}</i>', unsafe_allow_html=True)


def deepsearch_query():
    data = app.query(search_box)
    return data


def deepsearch_s3_add():
    return app.add_data(s3_url)


def deepsearch_local_add():
    return app.add_data(local_path)


local_css("style.css")
remote_css("https://fonts.googleapis.com/icon?family=Material+Icons")

# Create a list of tab titles
tab_titles = ["Add", "Query"]

# Create a tabbed layout
tabs = st.tabs(tab_titles)

# Add content to the first tab
with tabs[0]:
    option = st.selectbox("Choose the source", ("Local", "S3"))
    if option == "Local":
        local_path = st.text_input("local_path")
        if st.button("add_local"):
            print("Adding local path {}...".format(local_path))
            response = deepsearch_local_add()
    else:
        s3_url = st.text_input("S3 URL")
        if st.button("add_s3"):
            print("Adding S3 URL {}...".format(s3_url))
            response = deepsearch_s3_add()

# Add content to the second tab
with tabs[1]:
    search_box = st.text_input("search_box", "")

    # Write the response to the UI.
    if st.button("🔍 Search", key="search_button"):
        print("Searching {}...".format(search_box))
        response = deepsearch_query()
        st.write("Response:")
        st.write(response)
