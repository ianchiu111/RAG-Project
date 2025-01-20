# Reference：https://smith.langchain.com/onboarding?organizationId=07acc26b-a455-4c66-8557-971fd98e30c4&step=1
# （Stop Working on it）Topic：Build a semantic search engine over a PDF with document loaders, embedding models, and vector stores

import os
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage

# Change false to true when needed
os.environ["LANGSMITH_TRACING"] = "false" 
LANGSMITH_API_KEY = os.getenv('LANGSMITH_API_KEY')
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')

# Openai api with Langchain Framework
# setup model's parameter, like Temperature, N, top_p, etc
llm = ChatOpenAI(
    model = "gpt-4-turbo",
    temperature = 1.2,
    max_tokens = 20,
)

# # Try to load PDF files
from langchain_community.document_loaders import PyPDFLoader

# file_path = ("C:/論文紀錄/12_The Rise of Artificial Intelligence under the Lens of Sustainability.pdf")
# loader = PyPDFLoader(file_path)

# docs = loader.load()
# print(docs[0])

# Try to load Web Pages
import bs4
import asyncio
from langchain_community.document_loaders import WebBaseLoader

page_url = "https://python.langchain.com/docs/how_to/chatbots_memory/"

loader = WebBaseLoader(web_paths=[page_url])
docs = []

# 異步函數來處理 alazy_load
## async for 需要在協程（由 async def 定義的函數）中運行
async def load_documents(file_path):
    loader = PyPDFLoader(file_path)
    docs = []
    async for doc in loader.alazy_load():
        docs.append(doc)
    return docs

# 使用 asyncio.run() 執行異步函數
docs = asyncio.run(load_documents(page_url))

