import os

from dotenv import load_dotenv

from core.database import DB

load_dotenv()

CHROMA_PATH = os.getenv('CHROMA_PATH')


def query_rag(query_text: str) -> str:
    """Query the RAG model with the given text."""

    retriever = DB.as_retriever(search_kwargs={'k': 6})

    docs = retriever.invoke(query_text)

    # result = DB.similarity_search_with_score(query_text, k=5)

    for res in docs:
        print(res)
        print('\n')
    return docs


query_rag('abertura capitulo estudantil')
