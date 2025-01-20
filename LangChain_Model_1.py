# Reference：https://smith.langchain.com/onboarding?organizationId=07acc26b-a455-4c66-8557-971fd98e30c4&step=1
#（Stop Working on it）Topic：Build a simple LLM application with prompt templates and chat models

import os
import time
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
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

# How to Partially Format Prompt Templates
## prompt_template.partial("內部放值") 值可放 string 或透過 function 呼叫取得

## Partial with strings
from langchain_core.prompts import PromptTemplate

prompt_template_2 = PromptTemplate.from_template("{one}{two}")

prompt_template_2 = prompt_template_2.partial(one="Hello\t")
prompt_template_2 = prompt_template_2.partial(two="World!")

final_prompt_1 = prompt_template_2.format()

print(final_prompt_1)

## Partial with functions
from datetime import datetime

def get_datetime():
    now = datetime.now()
    return now.strftime("%m/%d/%Y, %H:%M:%S")

prompt_template_3 = PromptTemplate(
    template="Tell me a {adjective} joke about the day {date}",
    input_variables=["adjective", "date"],
)
prompt_template_3 = prompt_template_3.partial(date = get_datetime)
final_prompt_2 = prompt_template_3.format(adjective="funny")

print(final_prompt_2)

# Chat Prompt Composition
prompt_1 = SystemMessage(content="You are a nice pirate")
prompt_2 = (
    prompt_1 + HumanMessage(content="hi") + AIMessage(content="what?") + "{input}"
)
prompt_3 =prompt_2.format_messages(input="i said hi")
print(prompt_3)

# Pipeline Prompt
from langchain_core.prompts import PipelinePromptTemplate

full_prompt_template = PromptTemplate.from_template(
    """ 
    {introduction}

    {example}
    
    {start}
    """
)

introduction_prompt_template = PromptTemplate.from_template(
    """You are impersonating {person}."""
)

example_prompt_template = PromptTemplate.from_template(
    """
    Here's an example of an interaction:

    Q: {example_q}  
    A: {example_a}
    """
)

start_prompt_template = PromptTemplate.from_template(
    """
    Now, do this for real!

    Q: {input}
    A:
    """
)

input_prompts_template = [
    ("introduction", introduction_prompt_template),
    ("example", example_prompt_template),
    ("start", start_prompt_template),
]

pipeline_prompt = PipelinePromptTemplate(
    final_prompt = full_prompt_template, 
    pipeline_prompts = input_prompts_template
)

pipeline_prompt.input_variables

print(
    pipeline_prompt.format(
        person = "Elon Musk",
        example_q = "What's your favorite car?",
        example_a = "Tesla",
        input = "What's your favorite social media site?",
    )
)



























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