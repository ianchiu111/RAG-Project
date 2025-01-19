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

#Try to use few-shot examples with Prompt Template（ChatPromptTemplate.from_template()）

prompt_template_2 = ChatPromptTemplate.from_template(
    "Question: {question} \n {answer}"
)

## Example sets to the formatter prompt
examples = [
    {
        "question": "Who was the maternal grandfather of George Washington?",
        "answer": 
        """
        Are follow up questions needed here: Yes.
        Follow up: Who was the mother of George Washington?
        Intermediate answer: The mother of George Washington was Mary Ball Washington.
        Follow up: Who was the father of Mary Ball Washington?
        Intermediate answer: The father of Mary Ball Washington was Joseph Ball.
        So the final answer is: Joseph Ball
        """,
    },
    {
        "question": "Are both the directors of Jaws and Casino Royale from the same country?",
        "answer": 
        """
        Are follow up questions needed here: Yes.
        Follow up: Who is the director of Jaws?
        Intermediate Answer: The director of Jaws is Steven Spielberg.
        Follow up: Where is Steven Spielberg from?
        Intermediate Answer: The United States.
        Follow up: Who is the director of Casino Royale?
        Intermediate Answer: The director of Casino Royale is Martin Campbell.
        Follow up: Where is Martin Campbell from?
        Intermediate Answer: New Zealand.
        So the final answer is: No
        """,
    },
]
''' Generation Part
# Different way to print out 
## Print out like a prompt and response
prompt = prompt_template_2.invoke(examples[0])
response = llm.invoke(prompt)
print(response.content)
## Print out like QA storytelling 
print(prompt_template_2.invoke(examples[0]).to_string())
'''


# Static Few-Shot Examples in Chat Models
'''
# 較差的 few-shot Prompting Model
## Try to Pass the examples and formatter to FewShotPromptTemplate
from langchain_core.prompts import PromptTemplate, FewShotPromptTemplate

prompt_template_3 = PromptTemplate(
    input_variables=["question", "answer"],
    template="Question: {question}\n{answer}"
)

prompt = FewShotPromptTemplate(
    examples = examples,
    example_prompt = prompt_template_3,
    suffix = "Question: {input}",
    input_variables = ["input"],
)

## FewShotPromptTemplate have to use PromptTemplate Paackage
print(prompt.invoke({"input": "Who was the father of Mary Ball Washington?"}).to_string())
'''

from langchain_core.prompts import FewShotChatMessagePromptTemplate

llm_2 = ChatOpenAI(model="gpt-4-turbo")

# Few-Shot Examples
few_shot_examples = [
    {"input": "2 🦜 2", "output": "4"},
    {"input": "2 🦜 3", "output": "5"},
]

# Normal Prompt Template to format each example above.
prompt_template = ChatPromptTemplate.from_messages(
    [
        ("user", "{input}"),
        ("system", "{output}"),
    ]
)

# Define Few-Shot Prompt Template
## Must to combine ChatPromptTemplate or chain to make final prompt
few_shot_prompt = FewShotChatMessagePromptTemplate(
    example_prompt = prompt_template,
    examples = few_shot_examples,
)

# Combine to make final prompt
final_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "You are a wondrous wizard of math and answer it."),
        few_shot_prompt,
        ("user", "{input}"),
    ]
)

# chain = final_prompt | llm_2 是一種 pipeline 的概念，將 final_prompt 的輸出直接接到 llm_2
# final_prompt 負責：把 system 訊息、few-shot examples、user 的 input 一起組合成最終完整的 Chat 訊息列表。
chain = final_prompt | llm_2

# Different Results between normal text response and few-shot 
response = chain.invoke({"input": "What is 2 🦜 9?"})
print(response.content)

''' 錯誤範例
此方法模型無法理解 few-shot 的定義
response = llm_2.invoke("What is the result of 2 🦜 9?")
print(response.content)
'''

# Dynamic Few-Shot Examples in Chat Models
## Replace the (multiple types of) examples passed into FewShotChatMessagePromptTemplate with an example_selector
## Before install Chroma, make sure C++ environment is established
### Install with <https://visualstudio.microsoft.com/zh-hant/visual-cpp-build-tools/>
from langchain_chroma import Chroma
from langchain_core.example_selectors import SemanticSimilarityExampleSelector
from langchain_openai import OpenAIEmbeddings

examples = [
    # Examples type 1
    {"input": "2 🦜 2", "output": "4"},
    {"input": "2 🦜 3", "output": "5"},
    {"input": "2 🦜 4", "output": "6"},

    # Examples type 2
    {"input": "What did the cow say to the moon?", "output": "nothing at all"},
    
    # Examples type 3
    {
        "input": "Write me a poem about the moon",
        "output": "One for the moon, and one for me, who are we to talk about the moon?",
    },
]

to_vectorize = [" ".join(example.values()) for example in examples]
embeddings = OpenAIEmbeddings()
vectorstore = Chroma.from_texts(to_vectorize, embeddings, metadatas=examples)

example_selector = SemanticSimilarityExampleSelector(
    vectorstore=vectorstore,
    k=2,
)

# The prompt template will load examples by passing the input do the `select_examples` method
example_selector.select_examples({"input": "horse"})














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