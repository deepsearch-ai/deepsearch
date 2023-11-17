import streamlit as st

from deepsearchai.app import App
from deepsearchai.enums import MEDIA_TYPE

app = App()


def local_css(file_name):
    with open(file_name) as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)


def remote_css(url):
    st.markdown(f'<link href="{url}" rel="stylesheet">', unsafe_allow_html=True)


def icon(icon_name):
    st.markdown(f'<i class="material-icons">{icon_name}</i>', unsafe_allow_html=True)


def deepsearch_query():
    selected_multimedia = []
    if "AUDIO" in selected_datasources:
        selected_multimedia.append(MEDIA_TYPE.AUDIO)
    if "IMAGE" in selected_datasources:
        selected_multimedia.append(MEDIA_TYPE.IMAGE)
    if "VIDEO" in selected_datasources:
        selected_multimedia.append(MEDIA_TYPE.VIDEO)
    data = app.query(search_box, selected_multimedia, n_results)
    st.write("Response", data.get("llm_response"))
    st.write("Audio Matches", data.get("documents").get(MEDIA_TYPE.AUDIO))
    st.write("Image Matches", data.get("documents").get(MEDIA_TYPE.IMAGE))
    st.write("Video Matches", data.get("documents").get(MEDIA_TYPE.VIDEO))


def deepsearch_s3_add():
    return app.add_data(s3_url)


def deepsearch_local_add():
    return app.add_data(local_path)


remote_css("https://fonts.googleapis.com/icon?family=Material+Icons")

# Create a list of tab titles
tab_titles = ["Add", "Query"]

# Create a tabbed layout
tabs = st.tabs(tab_titles)

# Add content to the first tab
with tabs[0]:
    option = st.selectbox("Choose the source", ("Add Local", "Add S3"))
    if option == "Add Local":
        local_path = st.text_input("Provide local path for a file/folder")
        if st.button("Add"):
            print("Adding local path {}...".format(local_path))
            response = deepsearch_local_add()
    else:
        s3_url = st.text_input("Provide S3 URL for a bucket/file/folder")
        if st.button("Add S3"):
            print("Adding S3 URL {}...".format(s3_url))
            response = deepsearch_s3_add()

# Add content to the second tab
with tabs[1]:
    search_box = st.text_input("search_box", "")
    n_results = st.number_input("Number of Results of each Multimedia type", 1)
    datasources = ["AUDIO", "IMAGE", "VIDEO"]
    selected_datasources = st.multiselect(
        "Select datasources to be queried:", datasources
    )
    # Write the response to the UI.
    if st.button("🔍 Search", key="search_button"):
        print("Searching {}...".format(search_box))
        response = deepsearch_query()
        st.write("Response:")
        st.write(response)
