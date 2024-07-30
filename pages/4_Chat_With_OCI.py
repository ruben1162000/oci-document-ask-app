import streamlit as st

from genai_backend import *
from streamlit_js_eval import streamlit_js_eval

st.set_page_config(layout="wide",page_title="Search Engine OCI")


def refresh_page():
    streamlit_js_eval(js_expressions="parent.window.location.reload()")

st.header("An owl")
st.image("https://static.streamlit.io/examples/owl.jpg", width=200)