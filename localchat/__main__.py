from argparse import ArgumentParser
from json import load
from chat.console import ConsoleChat
from langchain.llms import GPT4All
from langchain.prompts import PromptTemplate
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler


if __name__ == "__main__":
    template = """You are a virtual assistant that help users answering \
        questions.
    Answer the QUESTION

    QUESTION
    {question}

    ANSWER
    """
    callbacks = [StreamingStdOutCallbackHandler()]
    prompt = PromptTemplate(
        template=template,
        input_variables=["question"]
    )

    parser = ArgumentParser()
    parser.add_argument(
        '--model',
        help='Path to model settings json.'
    )
    args = parser.parse_args()

    if not args.model:
        raise ValueError("Model path is required")

    with open(args.model, 'r') as f:
        model_settings = load(f)
        bin_path = model_settings['bin_path']
        backend = model_settings['backend']

    model = GPT4All(
        model=bin_path,
        backend=backend,
        callbacks=callbacks
    )

    chat = ConsoleChat(model=model, prompt=prompt)

    chat.start()
