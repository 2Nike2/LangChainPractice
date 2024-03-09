# これを実行して
# http://localhost:8000/agent/playground/
# にアクセスすると、Langchainのエージェントを使った対話ができる。

# 各種ライブラリ
# pip install langchain==0.1.4
# pip install langchain-openai==0.0.5
# pip install langchainhub==0.1.14
# pip install langserve[all]==0.0.47
# pip install pydantic==1.10.13

from typing import List
from fastapi import FastAPI
from langchain_openai import ChatOpenAI
from langchain_community.document_loaders import WebBaseLoader
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.tools.retriever import create_retriever_tool
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain import hub
from langchain.agents import create_openai_functions_agent
from langchain.agents import AgentExecutor
from langchain.pydantic_v1 import BaseModel, Field
from langchain_core.messages import BaseMessage
from langserve import add_routes

# リトリーバの設定
loader = WebBaseLoader("https://www.aozora.gr.jp/cards/000081/files/43754_17659.html")
docs = loader.load()
text_splitter = RecursiveCharacterTextSplitter()
documents = text_splitter.split_documents(docs)
embeddings = OpenAIEmbeddings()
vector = FAISS.from_documents(documents, embeddings)
retriever = vector.as_retriever()

# ツールの設定
retriever_tool = create_retriever_tool(
    retriever,
    'miyazawakenji_restaulant',
    '宮沢賢治の「注文の多い料理店」について調べるツールです。宮沢賢治の「注文の多い料理店」に関する質問の場合はこのツールを使って答えて下さい。'
)

search_tool = TavilySearchResults()

tools = [retriever_tool, search_tool]

# エージェントの作成
prompt = hub.pull("hwchase17/openai-functions-agent")
llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)
agent = create_openai_functions_agent(llm, tools, prompt)
agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)


# Webアプリの定義
app = FastAPI(
    title="Langchain Server",
    version="1.0",
    description="A simple API server using LangChain's Runnable interfaces"
)

# 入出力定義
class Input(BaseModel):
    input: str
    chat_history: List[BaseMessage] = Field(
        ...,
        extra={"widget": {"type": "chat", "input": "location"}}
    )

class Output(BaseModel):
    output: str

add_routes(
    app,
    agent_executor.with_types(input_type=Input, output_type=Output),
    path="/agent"
)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="localhost", port=8000)

