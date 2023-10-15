from .base import BaseIngestor
import os
from typing import List

from langchain.embeddings.gpt4all import GPT4AllEmbeddings

from langchain.schema.embeddings import Embeddings

from langchain.document_loaders import TextLoader
from langchain.docstore.document import Document

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings.sentence_transformer import SentenceTransformerEmbeddings
from langchain.vectorstores import Chroma


class ChromaIngestor(BaseIngestor):
    _raw_files_folder: str
    _output_folder: str
    embeddings_model: Embeddings
    _db: Chroma

    def __init__(self, files_folder: str, output_folder: str):
        self._raw_files_folder = files_folder
        self._output_folder = output_folder
        # self._embeddings_model = embedding_model
    
    def load_file(self, file_path: str) -> Document:
        loader = TextLoader(file_path)
        return loader.load()[0]

    def load_files(self) -> List[Document]:
        files = [os.path.join(dirpath,f) for (dirpath, dirnames, filenames) in os.walk(self._raw_files_folder) for f in filenames]
        documents = [self.load_file(f) for f in files]
        return documents
    
    def split_documents(self, documents: List[Document]) -> List[Document]:
        splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        chunks = splitter.split_documents(documents)
        return chunks

    def ingest(self):
        # create the open-source embedding function
        embedding_function = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")
        db = Chroma.from_documents(
            documents=chunks,
            persist_directory=self.output_folder
        )
