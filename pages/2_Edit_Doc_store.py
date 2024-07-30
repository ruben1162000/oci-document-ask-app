import streamlit as st

from genai_backend import *
from streamlit_js_eval import streamlit_js_eval

st.set_page_config(layout="wide",page_title="Edit Document Store")


def refresh_page():
    streamlit_js_eval(js_expressions="parent.window.location.reload()")

st.header("EDIT DOC STORES")
doc_stores_df = get_document_stores()
doc_store_name_select = st.selectbox("Select Doc Store",label_visibility='hidden',options=doc_stores_df["DOC_STORE"])
doc_store_name,doc_store_description,is_search_store = doc_stores_df[doc_stores_df['DOC_STORE']==doc_store_name_select].values[0]
doc_store_description_new = st.text_area("Description",value=doc_store_description,key="desc_update")
is_search_store_new = st.checkbox("Search Engine Store",value=is_search_store,key="is_search_store_update")    
col1,col2,col3 = st.columns(3)
with col1:
    if st.button("Edit",type="secondary"):
        edit_document_store(doc_store_name,doc_store_description_new,is_search_store_new)  
        refresh_page()          
with col2:
    if st.button("Truncate Store",type="primary"):
        truncate_document_store(doc_store_name)
        refresh_page()
with col3:
    if st.button("Delete",type="primary"):
        delete_document_store(doc_store_name)
        refresh_page()


st.subheader("DELETE SOURCES")
col3,col4 = st.columns(2)
with col3:
    sources_selected = st.multiselect("Select Sources",options=get_document_store_sources(doc_store_name_select),label_visibility='collapsed')
with col4:
    if st.button("Remove Source(s)",type="primary"):
        delete_sources_from_document_store(doc_store_name,sources_selected)
        refresh_page()