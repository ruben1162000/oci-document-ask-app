import streamlit as st

from genai_backend import *
from streamlit_js_eval import streamlit_js_eval

# st.set_page_config(layout="wide",page_title="Q&A With OCI")


def refresh_page():
    streamlit_js_eval(js_expressions="parent.window.location.reload()")
# st.header("OCI Ask")
doc_stores_df = get_chat_document_stores()
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