from langchain.chains.summarize import load_summarize_chain
from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader#, WebBaseLoader
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
import gradio as gr
import os
load_dotenv()
api_key = os.getenv("openai_api_key")

def summary(file_paths):
  #loader = WebBaseLoader(web_path)
  # loader = DirectoryLoader(file_patob="**/*.md")
  llm = ChatOpenAI(temperature=0, model_name="gpt-3.5-turbo", api_key=api_key)
  chain = load_summarize_chain(llm, chain_type="stuff")
  docs = ""
  for path in file_paths:
    head = os.path.split(path)
    loader = PyPDFLoader(path)
    doc = loader.load()
    docs += str(head[1])+": "+str(chain.run(doc))+"\n\n"
  return docs

summ = gr.Interface(
    summary,
    [
      gr.File(label="Files", file_count="multiple")
    ],
    "textbox",
    title="Document Summarization using OpenAI's GPT-4 and Langchain 🦜",
    theme = "gradio/monochrome"
)
summ.launch()

