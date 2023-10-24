from langchain.document_loaders import DirectoryLoader
from langchain.embeddings import GPT4AllEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores.chroma import Chroma

PERSIST_DIRECTORY = "./persist"


def load_chunks(documents_path: str):
    """load splitted documents from a directory"""
    loader = DirectoryLoader(documents_path, glob="**/*.txt")
    raw_documents = loader.load()
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000)
    docs = splitter.split_documents(raw_documents)

    return docs


if __name__ == "__main__":
    embeddings_function = GPT4AllEmbeddings()
    chunks = load_chunks("./documents")
    db = Chroma.from_documents(
        chunks, embeddings_function, persist_directory=PERSIST_DIRECTORY
    )
    print("Done!")
