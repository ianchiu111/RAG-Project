'''
# Build a Chat Model --- Tool Calling with Agent
'''

import os
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain.agents import AgentExecutor, create_tool_calling_agent, tool

os.environ["LANGSMITH_TRACING"] = "true"  # Change false to true when needed
LANGSMITH_API_KEY = os.getenv('LANGSMITH_API_KEY')
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')

llm = ChatOpenAI(
    model = "gpt-4-turbo",
    temperature = 1.2,
    # logprobs = True 用於產生 log probability（較新的 LLM 不支援 logprobs 參數）
)

prompt_template = ChatPromptTemplate.from_messages(
    [
        ("user", "{input}"),
        # ("placeholder", "{agent_scratchpad}") open with agent part
    ]
)

# Topic 1：Function/Tool Calling / Combine with Agent（超級好用）
## 使用 tool calling 需要結合 Agent 代理功能
## Reference：https://python.langchain.com/api_reference/langchain/agents/langchain.agents.tool_calling_agent.base.create_tool_calling_agent.html

''' <Tool Calling + Agent Part>
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
#  Bonus：Topic 2：In Memory Cache
# -----------------------------
from langchain_core.globals import set_llm_cache
from langchain_core.caches import InMemoryCache
set_llm_cache(InMemoryCache())
# -----------------------------

# -----------------------------
#  Topic 3：Response MetaData
## Close agent part
# -----------------------------

# Standard Output of LLM
## 可以透過 print(response) 了解需要的詳細資料
prompt_template = prompt_template.invoke({"input": "what is 5 +3 and 5 * 3"})
response = llm.invoke(prompt_template)
print(response)

# -----------------------------
