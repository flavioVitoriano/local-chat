from .base import BaseChat
from langchain.chains import LLMChain
from langchain.llms import GPT4All, BaseLLM
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.prompts import PromptTemplate



class ConsoleChat(BaseChat):
    def start(self):
        chain = LLMChain(prompt=self._prompt, llm=self._model)

        while True:
            question = input("\n\nSay something: ")
            chain.run(question)
