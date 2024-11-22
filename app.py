import streamlit as st
from streamlit_feedback import streamlit_feedback
from langchain_core.messages import AIMessage, HumanMessage

from src.config.assistant_config import IeeeAssistant
from src.core.database import add_to_chroma, clear_database
from src.core.loader import load_pdf_directory, split_documents

if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []


def reset_chat():
    st.session_state.pop('chat_history', None)


def populate_database():
    documents = load_pdf_directory()
    chunks = split_documents(documents)
    add_to_chroma(chunks)


def get_response(user_input, chat_history=st.session_state.chat_history):
    if 'assistant' not in st.session_state:
        st.session_state.assistant = IeeeAssistant()
        st.session_state.assistant.get_assistant()
    return st.session_state.assistant.run_assistant(user_input, chat_history)


# Configuração da página do Streamlit
st.set_page_config(
    page_title='Iracema.AI', page_icon='💬', layout='centered'
)
st.title('💬 Fale com a Iracema.IA')
st.sidebar.title('Configurações')
st.sidebar.button('Resetar Chat', on_click=reset_chat)


for message in st.session_state.chat_history:
    if isinstance(message, AIMessage):
        with st.chat_message('AI', avatar="assets/carcarA4.png"):
            st.write(message.content)
    elif isinstance(message, HumanMessage):
        with st.chat_message('Human'):
            st.write(message.content)

user_query = st.chat_input('Digite sua mensagem:', key='user_input')

if user_query is not None and user_query != '':
    st.session_state.chat_history.append(HumanMessage(content=user_query))

    with st.chat_message('Human'):
        st.markdown(user_query)

    with st.spinner('Pensando...'):
        with st.chat_message('AI',avatar="assets/carcarA4.png"):
            stream = get_response(user_query, st.session_state.chat_history)
            response = st.write_stream(stream)

    st.session_state.chat_history.append(AIMessage(content=response))
