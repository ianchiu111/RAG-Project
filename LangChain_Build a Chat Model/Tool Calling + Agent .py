'''
# Build a Chat Model --- Tool Calling with Agent
'''

import os
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain.agents import AgentExecutor, create_tool_calling_agent, tool

# Change false to true when needed
os.environ["LANGSMITH_TRACING"] = "true" 
LANGSMITH_API_KEY = os.getenv('LANGSMITH_API_KEY')
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')

# Openai api with Langchain Framework
# setup model's parameter, like Temperature, N, top_p, etc
llm = ChatOpenAI(
    model = "gpt-4-turbo",
    temperature = 1.2,
    logprobs = True # 用於產生 log probability
)

prompt_template = ChatPromptTemplate.from_messages(
    [
        ("user", "{input}"),
        ("placeholder", "{agent_scratchpad}")
    ]
)

# Topic 1：Function/Tool Calling / Combine with Agent（超級好用）
## 使用 tool calling 需要結合 Agent 代理功能
## Reference：https://python.langchain.com/api_reference/langchain/agents/langchain.agents.tool_calling_agent.base.create_tool_calling_agent.html
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
agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)

agent_executor.invoke({"input": "what is the value of magic_function(3)?"})

# -----------------------------
#  Bonus：Topic 2：In Memory Cache
# -----------------------------
from langchain_core.globals import set_llm_cache
from langchain_core.caches import InMemoryCache
set_llm_cache(InMemoryCache())
# -----------------------------


