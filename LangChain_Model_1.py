# Reference：https://smith.langchain.com/onboarding?organizationId=07acc26b-a455-4c66-8557-971fd98e30c4&step=1
# Topic：Build a simple LLM application with prompt templates and chat models

import os
import time
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate

# Change false to true when needed
os.environ["LANGSMITH_TRACING"] = "true" 
LANGSMITH_API_KEY = os.getenv('LANGSMITH_API_KEY')
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')

# Openai api with Langchain Framework
# setup model's parameter, like Temperature, N, top_p, etc
llm = ChatOpenAI(
    model = "gpt-4-turbo",
    temperature = 1.2,
    max_tokens = 20,
)

## Message Types： SystemMessage, HumanMessage
messages = [
    SystemMessage("Start a dialogue with me"),
    HumanMessage("Introduce Tainan's weather."),
]

''' Generation Part
response = llm.invoke(messages)
'''

# Define Streaming Function
def stream_print_out(response):
    for chunk in response:
        print(chunk, end="", flush=True)
        time.sleep(0.05)  # 控制輸出速度

'''  Generation Part
# Comparasion about the response
## General Print Out
print("Response Content：", response.content)
## Streaming Print Out
stream_print_out(response.content)
'''

#Try to print out with Prompt Template（ChatPromptTemplate.from_messages()）

from langchain_core.prompts import ChatPromptTemplate

#system_template = "Translate the following from English into {language}"
prompt_template_1 = ChatPromptTemplate.from_messages(
    [
        ("system", "Translate the following from English into {language}"),
        ("user", "{text}")
    ]
)

# It's available to use invoke with {sentense} to make prompt completed, so we can make it flexible in application
## example：please enter the text and language you want to tranlate into
## user A：{language}：Chinese, {text}：Prompt Engineer.
## user B：{language}：Italian, {text}：Hello!
prompt = prompt_template_1.invoke({"language": "Italian", "text": "hi!"})
''' Generation Part
response = llm.invoke(prompt)
print(response.content)
'''

'''
# Define tool calling
# 尚未完成（暫時擱置）
def add(a: int, b: int) -> int:
    """Add two integers.

    Args:
        a: First integer
        b: Second integer
    """
    return a + b


def multiply(a: int, b: int) -> int:
    """Multiply two integers.

    Args:
        a: First integer
        b: Second integer
    """
    return a * b

tools = [add, multiply]

llm_with_bind_tool = llm.bind_tools(tools)
messages = [
    HumanMessage("(3 * 5) + 9"),
]

response = llm.invoke(messages)
print(response.content)
'''