
import streamlit as st

from genai_backend import *
from streamlit_js_eval import streamlit_js_eval

# st.set_page_config(layout="wide",page_title="Upload Chat Store Docs")


def refresh_page():
    streamlit_js_eval(js_expressions="parent.window.location.reload()")
# st.header("Upload Chat Store Docs")
doc_stores_df = get_chat_document_stores()
col1,col2 = st.columns(2)
with col1:
    doc_store_name_select = st.selectbox("Select Document Store",options=doc_stores_df["DOC_STORE"],label_visibility='collapsed')
    doc_files = st.file_uploader(label="PDF/Image", accept_multiple_files=True,type=ALLOWED_FILE_TYPES)       
with col2:
    test = st.button("Push Document(s)",type="primary")
    chunk_size = st.number_input("Chunk Size",value=500)
    chunk_overlap = st.number_input("Chunk Overlap",value=20)
if test:
    if not doc_store_name_select:
        st.error("Please select a document store")
    elif not chunk_size:
        st.error("Please enter Chunk Size")
    elif not chunk_overlap:
        st.error("Please enter Chunk Overlap")
    elif not doc_files:
        st.error("Please upload image/pdf files")
    else:
        push_documents_to_chat_store(doc_store_name_select,doc_files,chunk_overlap=chunk_overlap,chunk_size=chunk_size)
st.table(get_document_store(doc_store_name_select))

