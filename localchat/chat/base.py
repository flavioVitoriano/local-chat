from abc import ABC, abstractmethod
from langchain.llms import BaseLLM
from langchain.prompts import PromptTemplate


class BaseChat(ABC):
    _model: BaseLLM
    _prompt: PromptTemplate

    def __init__(self, prompt: PromptTemplate, model: BaseLLM):
        self._model = model
        self._prompt = prompt

    @abstractmethod
    def start(self):
        pass