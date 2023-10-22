from argparse import ArgumentParser
from json import load
from chat.console import ConsoleChat
from langchain.llms import GPT4All
from langchain.prompts import PromptTemplate
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument(
        '--model',
        required=True,
        help='Path to model settings json.'
    )
    args = parser.parse_args()