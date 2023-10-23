from abc import ABC, abstractmethod
from langchain.llms import BaseLLM
from langchain.prompts import PromptTemplate
from langchain.schema.embeddings import Embeddings
from langchain.vectorstores.chroma import Chroma


class BaseChat(ABC):
    _model: BaseLLM
    _prompt: PromptTemplate

    def __init__(
        self,
        prompt: PromptTemplate,
        model: BaseLLM,
        chroma_db: Chroma,
        embeddings_function: Embeddings,
    ):
        self._model = model
        self._prompt = prompt
        self._chroma = chroma_db
        self._embeddings_function = embeddings_function

    @abstractmethod
    def start(self):
        pass
