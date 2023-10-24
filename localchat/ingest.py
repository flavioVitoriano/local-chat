from langchain.embeddings import GPT4AllEmbeddings
from langchain.document_loaders import DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores.chroma import Chroma
from argparse import ArgumentParser
from schemas.model import ChatModel
from pydantic import ValidationError
import json


def load_chunks(documents_path: str):
    """load splitted documents from a directory"""
    loader = DirectoryLoader(documents_path, glob="**/*.txt")
    raw_documents = loader.load()
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000)
    docs = splitter.split_documents(raw_documents)

    return docs


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--model", required=True, help="Path to model settings json.")
    args = parser.parse_args()

    with open(args.model, "r") as f:
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
