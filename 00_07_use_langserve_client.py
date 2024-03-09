# pip install langserve[all]==0.0.47

from langserve import RemoteRunnable

remote_chain = RemoteRunnable("http://localhost:8000/agent/")
print(remote_chain.invoke({
    "input": "「注文の多い料理店」で店が出した札に書いてあった日本語の店名は？",
    "chat_history": []
}))