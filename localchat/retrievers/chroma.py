from langchain.vectorstores import Chroma
from langchain.schema.embeddings import Embeddings


class ChromaRetriever:
    def __init__(self, persist_directory: str, embedding_function: Embeddings = None):
        embedding_function = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")
        self._db = Chroma(persist_directory=persist_directory, embedding_function=embedding_function)

    def get_documents(self, query: str):
        return self._db.similarity_search(query, k=2)
