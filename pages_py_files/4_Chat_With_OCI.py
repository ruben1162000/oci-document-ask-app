import streamlit as st,time
import uuid
from genai_backend import *
from streamlit_js_eval import streamlit_js_eval

# st.set_page_config(layout="wide",page_title="Chat With OCI",page_icon=":speech_balloon:")
page_height=streamlit_js_eval(js_expressions='parent.window.innerHeight',want_output = True)
chat_names = get_all_chat_names()
doc_stores_df = get_chat_document_stores()


def refresh_page():
    streamlit_js_eval(js_expressions="parent.window.location.reload()")

def generate_random_id():
    return str(uuid.uuid4())

# Function to add a new entry
def add_entry():
    new_chat = "chat_"+generate_random_id()
    st.session_state.current_chat=new_chat
    return new_chat  

def apply_chat(prompt,doc_store,model_id,chat_history_limit,context_retrieval_limit):
    return chat(st.session_state.current_chat,prompt,doc_store,model_id,chat_history_limit,context_retrieval_limit)    

def stream_ai_message(ai_message):
    for chunk in ai_message:
        yield chunk


if 'current_chat' not in st.session_state:
    if(chat_names):
        st.session_state.current_chat = chat_names[0]        
    else:
        add_entry()



col1,col2= st.columns([0.2,0.8])

# Sidebar layout
with col1:
    # Add new entry button      
    new_chat_btn = st.button('New Chat &#9998;',key="new chat button")
    if new_chat_btn:
        add_entry()        
    
    # List of chat entries
    with st.container(height=int(page_height*0.7)):
        for entry in chat_names:
            if st.button(entry,key=f"chat-key-{entry}",use_container_width=True):
                st.session_state.current_chat=entry
    
with col2:
    col2_1,col2_2,col2_3,col2_4 = st.columns([2,2,1,1])        
    with col2_1:
        doc_store =st.selectbox("Document Store",options=doc_stores_df["DOC_STORE"])
    with col2_2:
        model_id = st.selectbox("Select Chat Model",options=OCI_CHAT_MODELS,key="model_select_chat")
    with col2_3:
        chat_history_limit = st.number_input("Chat History Limit",value=4,min_value=0)
    with col2_4:
        context_retrieval_limit = st.number_input("Doc Context Limit",value=20,min_value=5)
    chat_box = st.container(height=int(page_height*0.6))          
    chat_messages = get_chat_messages(st.session_state.current_chat)
    with chat_box:
        for message in chat_messages:
            with st.chat_message(message.type):
                st.markdown(message.content)
    prompt = st.chat_input("Type Here....",key="user_chat_prompt")               
    if prompt:
        print(f"user: {prompt}")
        with chat_box:
            with st.chat_message('human'):
                st.markdown(prompt)            
            with st.chat_message('ai'):
                markdown_placeholder = st.empty()
                ai_response = apply_chat(prompt,doc_store,model_id,chat_history_limit,context_retrieval_limit)
                typed_text = ""
                for char in ai_response:
                    typed_text += char
                    markdown_placeholder.markdown(typed_text)
                    time.sleep(0.001)  # Adjust speed for typewriting effect
                # st.write_stream(stream_ai_message(ai_response))