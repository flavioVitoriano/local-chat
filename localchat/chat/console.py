from langchain.chains import LLMChain

from .base import BaseChat


class ConsoleChat(BaseChat):
    def start(self):
        chain = LLMChain(prompt=self._prompt, llm=self._model)

        while True:
            question = input("\n\nUser input: ")
            docs = self._chroma.similarity_search(question, k=2)
            docs_content = "\n".join([doc.page_content for doc in docs])

            result = chain.predict(question=question, documents=docs_content)
            print(result)
