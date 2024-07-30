import streamlit as st

from genai_backend import *
from streamlit_js_eval import streamlit_js_eval

st.set_page_config(layout="wide")


def refresh_page():
    streamlit_js_eval(js_expressions="parent.window.location.reload()")


tab1, tab2, tab3, tab4, tab5, tab6,tab7 = st.tabs(["Add Document Stores", "Edit Document Store" ,"PDF Search Engine OCI", "Chat With OCI", "Q&A with OCI", "Upload Chat Store Docs","Upload Search Store Docs"])

with tab1:
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
        elif not doc_store_description:
            st.error("Please enter doc store description")
        else:
            register_document_store(doc_store_name,doc_store_description,is_search_store)
        
    doc_stores_df = get_document_stores()
    st.table(doc_stores_df)


with tab2:
    st.header("EDIT DOC STORES")
    doc_stores_df = get_document_stores()
    doc_store_name_select = st.selectbox("Select Doc Store",label_visibility='hidden',options=doc_stores_df["DOC_STORE"])
    doc_store_name,doc_store_description,is_search_store = doc_stores_df[doc_stores_df['DOC_STORE']==doc_store_name_select].values[0]
    doc_store_description_new = st.text_area("Description",value=doc_store_description,key="desc_update")
    is_search_store_new = st.checkbox("Search Engine Store",value=is_search_store,key="is_search_store_update")    
    col1,col2 = st.columns(2)
    with col1:
        if st.button("Edit",type="primary"):
            edit_document_store(doc_store_name,doc_store_description_new,is_search_store_new)  
            refresh_page()          
    with col2:
        if st.button("Delete",type="secondary"):
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

            

with tab3:
    st.header("OCI Document Understanding Vector Based PDF Search Engine")
    col1,col2,col3 = st.columns([0.6,0.2,0.2])
    with col1:
        search_query = st.text_input("Doc Store Name",key="search_engine_query")

with tab4:
    st.header("An owl")
    st.image("https://static.streamlit.io/examples/owl.jpg", width=200)

with tab5:
    st.header("OCI Ask")
    doc_stores_df = get_document_stores()
    col1,col2 = st.columns(2)
    with col1:
        query = st.text_area("Ask Something....",placeholder="Ask Something....",key="query_qa",height=260,label_visibility='collapsed')
    with col2:
        doc_store_name_select = st.selectbox("Document Store",options=doc_stores_df["DOC_STORE"],label_visibility='collapsed')
        model_id = st.selectbox("select model",options=OCI_CHAT_MODELS+OCI_GEN_MODELS,key="model_select_qa")
        context_limit = st.number_input("Context Limit",value=30)
        test = st.button("Go!!",type="primary")
    if test:
        if not doc_store_name_select:
            st.error("Please select a document store")
        elif not context_limit:
            st.error("Please enter context limit")
        else:
            rag_result = get_rag_response(query,doc_store_name_select,model_id=model_id,context_limit=30)
            st.success(rag_result["result"])
            st.info("Relevant Texts")
            df  = pd.DataFrame(rag_result["source_documents"]).iloc[:, 0:-1].applymap(lambda x: x[1])
            df = df.drop(1,axis=1).join(pd.json_normalize(df[1])).rename(columns={0:"Text"})
            st.table(df)

with tab6:
    st.header("Upload Chat Store Docs")
    doc_stores_df = get_document_stores()
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

with tab7:
    
    st.image("https://static.streamlit.io/examples/dog.jpg", width=200)

