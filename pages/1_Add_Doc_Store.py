import streamlit as st

from genai_backend import *
from streamlit_js_eval import streamlit_js_eval

st.set_page_config(layout="wide",page_title="Add Document Store")


def refresh_page():
    streamlit_js_eval(js_expressions="parent.window.location.reload()")


st.header("ADD DOC STORES")
doc_store_name = st.text_input("Doc Store Name",key="name_add")

# Multi-line text input
doc_store_description = st.text_area("Description",key="desc_add")

# Checkbox
is_search_store = st.checkbox("Search Engine Store",key="is_search_store_add")
if is_search_store:
    st.info("Note that this uses OCI Document Understanding Service",icon="ℹ️")


# If you want to use the inputs for further processing, you can add more logic here
if st.button("Add Doc Store"):
    if not doc_store_name:
        st.error("Please enter doc store name")
    elif doc_store_name=="DOCUMENT_STORE_LIST":
        st.error("This name is reserved")
    elif not doc_store_description:
        st.error("Please enter doc store description")
    else:
        register_document_store(doc_store_name,doc_store_description,is_search_store)
    
doc_stores_df = get_document_stores()
st.table(doc_stores_df)
