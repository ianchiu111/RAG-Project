'''
# Build a Chat Model --- Tool Calling with Agent
## LangChain Version：version v0.3.17
'''

import os
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate

os.environ["LANGSMITH_TRACING"] = "true"  # Change false to true when needed
LANGSMITH_API_KEY = os.getenv('LANGSMITH_API_KEY')
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')

prompt_template = ChatPromptTemplate.from_messages(
    [
        ("user", "{input}"),
        #("placeholder", "{agent_scratchpad}") #open with agent part
    ]
)

# -----------------------------
#  Topic 4：Rate Limiter
## 目前為 Beta 階段，若有調整再做改變
# -----------------------------

from langchain_core.rate_limiters import InMemoryRateLimiter

rate_limiter = InMemoryRateLimiter(
    requests_per_second = 0.1,  # <-- Super slow! We can only make a request once every 10 seconds!!
    check_every_n_seconds = 0.1,  # Wake up every 100 ms to check whether allowed to make a request,
    max_bucket_size = 10,  # Controls the maximum burst size.
)
# -----------------------------

llm = ChatOpenAI(
    model = "gpt-4-turbo",
    temperature = 1.2,
    # logprobs = True 用於產生 log probability（較新的 LLM 不支援 logprobs 參數）
    rate_limiter=rate_limiter #結合 Topic 4：Rate Limiter
)

# Topic 1：<Function/Tool Calling> combine with <Agent>（超級好用）
## 使用 tool calling 需要結合 Agent 代理功能
## Reference：https://python.langchain.com/api_reference/langchain/agents/langchain.agents.tool_calling_agent.base.create_tool_calling_agent.html

''' <Tool Calling + Agent Part>
from langchain.agents import AgentExecutor, create_tool_calling_agent, tool

@tool
def add(a: int, b: int) -> int:
    """ 將兩數相加 """
    return a + b

@tool
def multiply(a: int, b: int) -> int:
    """ 將兩數相乘 """
    return a * b

tools = [add, multiply]

# -----------------------------
#  Initialize the Agent（以下為新版的 Agent 架構）
#  Run the Agent with a Query
# -----------------------------
# from langchain.agents import AgentExecutor, create_tool_calling_agent, tool

agent = create_tool_calling_agent(llm, tools, prompt_template)
agent_executor = AgentExecutor(
    agent = agent ,
    tools = tools ,
    verbose = False, # verbose = True 會啟用 LangChain 的日誌記錄功能 → 導致 StdOutCallbackHandler.on_chain_start callback: AttributeError("'NoneType' object has no attribute 'get'")"
)

response = agent_executor.invoke({"input": "what is 5 +3 and 5 * 3"})
print(response["output"])
'''

# -----------------------------
#  Topic 2：<In Memory Cache> combine with <Rate Limiter>（可以呈現出 In Memory Cache 的用處）
## Cache 功能可以減少相同題目重複詢問的狀況 → 降低模型回覆的時間成本與浪費
# -----------------------------
'''
from langchain_core.globals import set_llm_cache
from langchain_core.caches import InMemoryCache

set_llm_cache(InMemoryCache())

import time
for _ in range(5):
    start = time.time()
    llm.invoke("hello")
    end = time.time()
    print(end - start)
'''
# -----------------------------


# -----------------------------
#  Topic 3：Response MetaData
## 1. Must to close agent part
## 2. Standard Output of LLM to understand the details
# -----------------------------
prompt_template = prompt_template.invoke({"input": "what is 5 +3 and 5 * 3"})
response = llm.invoke(prompt_template)
print(response.content)
# -----------------------------




