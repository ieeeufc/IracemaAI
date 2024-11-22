from dotenv import load_dotenv

load_dotenv()
from src.config.assistant_config import IeeeAssistant
from src.core.database import add_to_chroma
from src.core.loader import load_pdf_directory, split_documents

LOAD = False

if __name__ == '__main__':
    if LOAD:
        documents = load_pdf_directory()
        print(f'Loaded {len(documents)} documents')
        print(documents[0])

        print('Splitting documents...')
        chunks = split_documents(documents)
        print(f'Split {len(documents)} documents into {len(chunks)} chunks')
        print(chunks[0])

        print('Done!')

        print('Adding chunks to Chroma...')
        add_to_chroma(chunks)
        print('Done!')

    assistant = IeeeAssistant()
    assistant.get_assistant()
    print('Assistant initialized!')

    while True:
        query = input('Ask me anything: ')
        resposta = assistant.run_assistant(query)
        print(resposta)
