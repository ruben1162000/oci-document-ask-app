
import streamlit as st

from genai_backend import *
from streamlit_js_eval import streamlit_js_eval


# st.set_page_config(layout="wide",page_title="Uplaod Search Store Docs")


def refresh_page():
    streamlit_js_eval(js_expressions="parent.window.location.reload()")


# st.header("Upload Search Store Docs")
doc_stores_df = get_search_document_stores()
col1,col2 = st.columns(2)
with col1:
    doc_store_name_select = st.selectbox("Select Document Store",options=doc_stores_df["DOC_STORE"],label_visibility='collapsed')
    doc_files = st.file_uploader(label="PDF/Image", accept_multiple_files=True,type=ALLOWED_FILE_TYPES)       
with col2:
    test = st.button("Initiate OCR",type="primary")
    job_name = st.text_input("OCI Doc Understanding Job Name")

if test:
    if not doc_store_name_select:
        st.error("Please select a document store")
    elif not job_name:
        st.error("Please enter OCR Job Name")   
    elif not doc_files:
        st.error("Please upload image/pdf files")
    else:
        intiate_search_store_ocr_job(doc_store_name_select,doc_files,job_name)
        # refresh_page()

col3,col4 = st.columns([0.9,0.1])
with col3:
    st.success("Documents In Store")        
with col4:
    hide_jobs = st.checkbox("Hide",value=True)
if not hide_jobs:
    st.table(get_document_store(doc_store_name_select))

col3,col4 = st.columns([0.9,0.1])
with col3:
    st.warning("OCR Jobs")        
with col4:
    hide_jobs = st.checkbox("Hide",value=False)
if not hide_jobs:
    st.table(list_search_store_ocr_jobs(doc_store_name_select).drop(columns=["FILES_SUBMITTED","FILES_PROCESSED"]))


col3,col4 = st.columns([0.7,0.3])
with col3:
    st.info("Document JSON Responses")        
with col4:
    process_json_files = st.button("Process JSON Files",type="primary")
if process_json_files:
    process_search_store_json(doc_store_name_select,list_search_store_ocr_json(doc_store_name_select))


for _,x in list_search_store_ocr_json(doc_store_name_select).T.items():
    col5,col6 = st.columns([0.7,0.3])
    with col5:
        st.success(x["json_file"])
    with col6:
        hide_json = st.checkbox("Show",value=False,key=f"chk_{x['json_file']}")
    if hide_json:
        st.json(x["content"],expanded=False)



