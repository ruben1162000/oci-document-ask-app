import streamlit as st

from genai_backend import *
from streamlit_js_eval import streamlit_js_eval
from streamlit_pdf_viewer import pdf_viewer
import io

st.set_page_config(layout="wide",page_title="Search Engine OCI")
fs.invalidate_cache()

def refresh_page():
    streamlit_js_eval(js_expressions="parent.window.location.reload()")

st.header("OCI Document Understanding Vector Based PDF Search Engine")
doc_stores_df = get_search_document_stores()
col1,col2= st.columns([0.8,0.2])
with col1:
    search_query = st.text_area("Search Query",placeholder="Type Something Here.....",key="search_engine_query",label_visibility='collapsed')
with col2:
    doc_store_name_select = st.selectbox("Select Document Store",options=doc_stores_df["DOC_STORE"],label_visibility='collapsed')
    btn_search = st.button("Search",type="primary")    
if btn_search and bool(search_query):
    docs = search_docs(search_query,doc_store_name_select,n_results=50)
    df  = pd.DataFrame(docs).iloc[:, 0:-1].applymap(lambda x: x[1])
    df = df.drop(1,axis=1).join(pd.json_normalize(df[1])).rename(columns={0:"text"})
    df["color"] = "blue"
    df["annotation_outline_size"]=10
    df = df.astype({"page":int,"width":int,"height":int,"x":int,"y":int})
    grouped=df.groupby(by="source")
    for group_name, group_data in grouped:
        pages_to_render=group_data["page"].unique()
        pages_to_render.sort()
        pages_to_render = [int(x) for x in pages_to_render]
        # file_path = f"oci://{DOC_OCR_BUCKET}@{NAMESPACE_NAME}/{group_name.split('/',8)[-1][:-5]}"
        file_name = group_name.split('/')[-1]
        st.info(file_name)
        pdf_viewer(fs.cat_file(group_name),annotations=(group_data.drop(columns=["text","source"]).to_dict(orient="records")),pages_to_render=pages_to_render)
        



