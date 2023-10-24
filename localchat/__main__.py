from argparse import ArgumentParser
from json import load
from chat.console import ConsoleChat
from schemas.model import ChatModel
from langchain.llms import GPT4All
from langchain.prompts import PromptTemplate
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.embeddings import GPT4AllEmbeddings
from langchain.vectorstores.chroma import Chroma
from ingest import PERSIST_DIRECTORY
from pydantic import ValidationError


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--model", required=True, help="Path to model settings json.")
    args = parser.parse_args()

    with open(args.model, "r") as f:
        model_arguments = load(f)

        try:
            model = ChatModel(**model_arguments)
        except ValidationError as e:
            print(e)
            exit(1)

    prompt = PromptTemplate(
        template=model.prompt_template, input_variables=model.input_variables
    )
    callbacks = [StreamingStdOutCallbackHandler()]
    model = GPT4All(model=model.bin_path, callbacks=callbacks)

    embedding_function = GPT4AllEmbeddings()
    db = Chroma(
        persist_directory=PERSIST_DIRECTORY, embedding_function=embedding_function
    )

    chat = ConsoleChat(
        model=model, prompt=prompt, chroma_db=db, embeddings_function=embedding_function
    )

    chat.start()
