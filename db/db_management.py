from src.core.database import clear_database, add_to_chroma
from src.core.loader import load_pdf_directory, split_documents
import time
def populate_database():
    documents = load_pdf_directory()
    chunks = split_documents(documents)
    add_to_chroma(chunks)
    print('Database populated!')

def clear_db():
    clear_database()
    print('Database cleared!')

if __name__ == '__main__':
    print('Clearing database...')
    print('This may take a while...')
    clear_db()
    time.sleep(5)
    clear_db()
    time.sleep(5)
    print('Populating database...')
    print('This may take a while...')
    populate_database()
    print('Done!')