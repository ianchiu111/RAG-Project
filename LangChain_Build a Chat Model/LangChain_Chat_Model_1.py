import os
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage
#from langchain_chroma import Chroma
#from langchain_openai import OpenAIEmbeddings
#from langchain_core.prompts import FewShotChatMessagePromptTemplate


# Change false to true when needed
os.environ["LANGSMITH_TRACING"] = "true" 
LANGSMITH_API_KEY = os.getenv('LANGSMITH_API_KEY')
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')

# Openai api with Langchain Framework
# setup model's parameter, like Temperature, N, top_p, etc
llm = ChatOpenAI(
    model = "gpt-4-turbo",
    temperature = 1.2,
)

prompt_template = ChatPromptTemplate.from_messages(
    [
        ("user", "{input}"),
        ("system", "{output}"),
    ]
)

# Topic 1：Function/Tool Calling / Combine with Agent（超級好用）
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
#  Initialize the Agent
# -----------------------------
from langchain.agents import initialize_agent, AgentType
agent = initialize_agent(
    tools=tools,
    llm=llm,
    agent=AgentType.OPENAI_FUNCTIONS,  # Choose appropriate agent type
    verbose=True  # Set to True for detailed logs
)
# -----------------------------
#  Run the Agent with a Query
# -----------------------------
query = "What is 3 * 12? Also, what is 11 + 49?"
response = agent.run(query)
print(f"Response: {response}")
