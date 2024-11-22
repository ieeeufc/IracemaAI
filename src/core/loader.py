from typing import List

from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_core.documents import Document
from langchain_experimental.text_splitter import SemanticChunker

from src.core.embeddings import get_embedding_function

PDFS_PATH = 'docs/'


load_dotenv()


# TODO: Add a google drive loader
def load_pdf_directory() -> List[Document]:
    """
    Loads all PDF documents from a specified directory.

    This function uses the PyPDFDirectoryLoader to load all PDF files
    from the directory specified by the PDFS_PATH constant.

    Returns:
        List[Document]: A list of Document objects representing the loaded
        PDF files.
    """
    loader = PyPDFDirectoryLoader(PDFS_PATH)
    return loader.load()


def split_documents(documents: List[Document]) -> List[Document]:
    """
    Splits a list of documents into smaller chunks using a semantic chunker.

    Args:
        documents (List[Document]): A list of Document objects to be split.

    Returns:
        List[Document]: A list of Document objects that have been split into
        smaller chunks.
    """

    splitter = SemanticChunker(get_embedding_function())
    # splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    return splitter.split_documents(documents)
