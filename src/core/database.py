import os
import shutil
from typing import List

from dotenv import load_dotenv
from langchain_chroma.vectorstores import Chroma
from langchain_core.documents import Document

from src.core.embeddings import get_embedding_function

load_dotenv()

CHROMA_PATH = 'chroma/'

class Database:
    def __init__(self):
        self.database = Chroma(
    persist_directory=CHROMA_PATH,
    embedding_function=get_embedding_function(),
)

def assign_chunk_ids(chunks: List[Document]) -> List[Document]:
    """
    Assigns unique chunk IDs to a list of Document chunks based on
    their source and page metadata.

    Each chunk ID is composed of the source, page, and an index that
    increments for chunks on the same page.

    Args:
        chunks (List[Document]): A list of Document objects, each containing
        metadata with 'source' and 'page' keys.

    Returns:
        List[Document]: The list of Document objects with updated metadata
        including unique 'id' keys.
    """
    last_page_id = None
    current_chunk_index = 0

    for chunk in chunks:
        source = chunk.metadata.get('source')
        page = chunk.metadata.get('page')
        current_page_id = f'{source}:{page}'

        if current_page_id == last_page_id:
            current_chunk_index += 1
        else:
            current_chunk_index = 0

        chunk_id = f'{current_page_id}:{current_chunk_index}'
        last_page_id = current_page_id

        chunk.metadata['id'] = chunk_id

    return chunks


def add_to_chroma(chunks: List[Document]):
    """
    Adds a list of Document chunks to the Chroma database if they do not
    already exist.

    Args:
        chunks (List[Document]): A list of Document objects to be added to the
        database.
    """
    db = Database()
    chunks_with_ids = assign_chunk_ids(chunks)
    existing_items = db.database.get(include=[])  # IDs are always included by default
    existing_ids = set(existing_items['ids'])
    print(f'Number of existing documents in DB: {len(existing_ids)}')

    new_chunks = []
    for chunk in chunks_with_ids:
        if 'id' in chunk.metadata and chunk.metadata['id'] not in existing_ids:
            new_chunks.append(chunk)

    if new_chunks:
        print(f'👉 Adding new documents: {len(new_chunks)}')
        new_chunk_ids = [chunk.metadata['id'] for chunk in new_chunks]
        db.database.add_documents(new_chunks, ids=new_chunk_ids)
    else:
        print('✅ No new documents to add')


def clear_database():
    """
    Deletes the directory specified by the CHROMA_PATH constant if it exists.

    This function checks if the directory at CHROMA_PATH exists. If it does,
    the directory and all its contents are removed using shutil.rmtree.

    Raises:
        OSError: If the directory cannot be removed.
    """
    if os.path.exists(CHROMA_PATH):
        shutil.rmtree(CHROMA_PATH)
