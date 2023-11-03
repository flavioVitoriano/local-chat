from langchain.embeddings import GPT4AllEmbeddings

from langchain.document_loaders import (
    DirectoryLoader,
    PyPDFLoader,
    TextLoader,
    UnstructuredHTMLLoader,
)

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores.chroma import Chroma

from argparse import ArgumentParser
from schemas.model import ChatModel
from pydantic import ValidationError

import json


def load_chunks(documents_path: str):
    """load splitted documents from a directory"""
    loaders = {
        ".pdf": DirectoryLoader(
            path=documents_path,
            glob="**/*.pdf",
            show_progress=True,
            loader_cls=PyPDFLoader,
        ),
        ".txt": DirectoryLoader(
            path=documents_path,
            glob="**/*.txt",
            show_progress=True,
            loader_cls=TextLoader,
        ),
        ".html": DirectoryLoader(
            path=documents_path,
            glob="**/*.html",
            show_progress=True,
            loader_cls=UnstructuredHTMLLoader,
        ),
    }

    raw_docs = []
    for loader in loaders.values():
        raw_docs.extend(loader.load())

    splitter = RecursiveCharacterTextSplitter(chunk_size=1000)
    docs = splitter.split_documents(raw_docs)

    return docs


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--model", required=True, help="Path to model settings json.")
    args = parser.parse_args()

    with open(args.model, "r", encoding="utf-8-sig") as f:
        model_arguments = json.load(f)

        try:
            model = ChatModel.model_validate(model_arguments)
        except ValidationError as e:
            print(e)
            exit(1)

    embeddings_function = GPT4AllEmbeddings()
    chunks = load_chunks(model.documents_directory_path)
    db = Chroma.from_documents(
        chunks, embeddings_function, persist_directory=model.persist_directory_path
    )
    print("Done!")
