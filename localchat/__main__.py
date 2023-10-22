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

    with open(args.model, 'r') as f:
        model_settings = load(f)
        bin_path = model_settings["bin_path"]
        prompt_template = model_settings["prompt_template"]
        input_variables = model_settings["input_variables"]

    prompt = PromptTemplate(
        template=prompt_template,
        input_variables=input_variables
    )
    callbacks = [StreamingStdOutCallbackHandler()]
    model = GPT4All(
        model=bin_path,
        callbacks=callbacks
    )

    chat = ConsoleChat(model=model, prompt=prompt)

    chat.start()
