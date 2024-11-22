# TODO: Improve the embeddings and add more models

from langchain_google_genai import GoogleGenerativeAIEmbeddings


def get_embedding_function() -> GoogleGenerativeAIEmbeddings:
    """
    Creates and returns an instance of GoogleGenerativeAIEmbeddings with a
    specified model.

    Returns:
        GoogleGenerativeAIEmbeddings: An instance of the
        GoogleGenerativeAIEmbeddings class initialized with the
        'embedding-004' model.
    """
    embeddings = GoogleGenerativeAIEmbeddings(
        model='models/text-embedding-004'
    )
    return embeddings
