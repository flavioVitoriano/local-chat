from abc import ABC, abstractmethod
from langchain.llms import BaseLLM


class BaseChat(ABC):
    @abstractmethod
    def start(self):
        pass