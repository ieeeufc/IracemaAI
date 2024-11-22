__import__('pysqlite3')
import sys

sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')

from datetime import datetime

import pandas as pd
import streamlit as st
from langchain_core.messages import AIMessage, HumanMessage
from streamlit_feedback import streamlit_feedback
from streamlit_gsheets import GSheetsConnection

from src.config.assistant_config import IeeeAssistant
from src.core.database import add_to_chroma
from src.core.loader import load_pdf_directory, split_documents

st.set_page_config(page_title='Iracema.AI', page_icon='💬', layout='centered')
st.title('💬 Fale com a Iracema.IA')
st.sidebar.title('Configurações')

def _get_session():
    from streamlit.runtime import get_instance
    from streamlit.runtime.scriptrunner import get_script_run_ctx

    runtime = get_instance()
    session_id = get_script_run_ctx().session_id
    session_info = runtime._session_mgr.get_session_info(session_id)
    if session_info is None:
        raise RuntimeError("Couldn't get your Streamlit Session object.")
    return session_id


if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []

if 'conn' not in st.session_state:
    st.session_state.conn = st.connection('gsheets', type=GSheetsConnection)


def current_time():
    return datetime.now().strftime('%Y-%m-%d-%H:%M:%S')


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

st.sidebar.button('Resetar Chat', on_click=reset_chat)


for message in st.session_state.chat_history:
    if isinstance(message, AIMessage):
        with st.chat_message('AI', avatar='assets/carcarA4.png'):
            st.write(message.content)
    elif isinstance(message, HumanMessage):
        with st.chat_message('Human'):
            st.write(message.content)

user_query = st.chat_input('Digite sua mensagem:', key='user_input')

if user_query is not None and user_query != '':
    st.session_state.chat_history.append(HumanMessage(content=user_query))

    with st.chat_message('Human'):
        st.markdown(user_query)

    with st.chat_message('AI', avatar='assets/carcarA4.png'):
        stream = get_response(user_query, st.session_state.chat_history)
        response = st.write_stream(stream)
    st.session_state.chat_history.append(AIMessage(content=response))

    streamlit_feedback(
        feedback_type='thumbs',
        optional_text_label='[Opcional] Por favor, forneça mais informações',
        key='feedback',
    )

if 'feedback' in st.session_state and st.session_state['feedback'] is not None:
    user_feedback = {
        'session_id': _get_session(),
        'inserted_at': current_time(),
        'user_message': st.session_state['chat_history'][-2].content,
        'assistant_message': st.session_state['chat_history'][-1].content,
        'feedback_score': 1
        if st.session_state['feedback']['score'] == '👍'
        else 0,
        'feedback_text': st.session_state['feedback']['text'],
    }

    actual = st.session_state.conn.read(worksheet='feedback', max_entries=None)
    update = pd.concat(
        [actual, pd.DataFrame(user_feedback, index=[0])], ignore_index=True
    )
    st.session_state.conn.update(worksheet='feedback', data=update)
    st.session_state.conn.reset()
