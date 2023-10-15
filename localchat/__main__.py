from chat.console import ConsoleChat
from langchain.chains import LLMChain
from langchain.llms import GPT4All
from langchain.prompts import PromptTemplate
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from ingestors.chroma import ChromaIngestor
from retrievers.chroma import ChromaRetriever

template = """You are a virtual assistant that help users answering questions.
Answer the QUESTION

QUESTION
{question}

ANSWER
"""
callbacks = [StreamingStdOutCallbackHandler()]
prompt = PromptTemplate(template=template, input_variables=["question", "docs"])
model = GPT4All(model="./bin/orca-mini-7b.ggmlv3.q4_0.bin", backend="orca", callbacks=callbacks)

chat = ConsoleChat(model=model, prompt=prompt)

chat.start()