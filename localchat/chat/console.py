from .base import BaseChat
from langchain.chains import LLMChain
from langchain.llms import GPT4All
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.prompts import PromptTemplate



class ConsoleChat(BaseChat):
    def start(self):
        template = """Question: {question}

Answer: Let's think step by step."""

        prompt = PromptTemplate(template=template, input_variables=["question"])
        model_path = "./models/orca-mini-7b.ggmlv3.q4_0.bin"
        callbacks = [StreamingStdOutCallbackHandler()]
        llm = GPT4All(model=model_path, callbacks=callbacks, backend="orca")
        chain = LLMChain(prompt=prompt, llm=llm)

        while True:
            question = input("\n\nSay something: ")
            chain.run(question)
        
