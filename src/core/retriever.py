import os

from dotenv import load_dotenv
from langchain_core.documents import Document

from src.core.database import Database

load_dotenv()

CHROMA_PATH = os.getenv('CHROMA_PATH')


class Retriever:
    """
    A class used to retrieve documents from a database using a RAG model.

    Attributes
    ----------
    database : DB
        An instance of the database to be used for retrieving documents.

    Methods
    -------
    query_rag(query_text: str) -> str
        Queries the RAG model with the given text and returns the retrieved
        documents.
    """

    def __init__(self):
        self.database = Database().database

    def query_rag(self, query_text: str) -> list[Document]:
        """
        Queries the retriever with the given query text and returns the
        retrieved documents.
        Args:
            query_text (str): The text to query the retriever with.
        Returns:
            str: The retrieved documents as a string. If an error occurs,
            returns None.
        """
        retriever = self.database.as_retriever(search_kwargs={'k': 6})

        try:
            docs = retriever.invoke(query_text)
        except Exception as e:
            print(f'An error occurred while invoking the retriever: {e}')
            docs = None

        return docs
