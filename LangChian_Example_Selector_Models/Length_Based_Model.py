'''
Length Based Example Selector Model
適用情境：
1. 限制 Prompt 的長度
2. Prompt 較短 → 選擇較多 Examples
3. Prompt 較長 → 選擇較少 Examples
'''

from langchain_core.example_selectors import LengthBasedExampleSelector
from langchain_core.prompts import FewShotPromptTemplate, PromptTemplate
import os
from langchain_openai import ChatOpenAI
# from langchain.chains import LLMChain 這個用法已遭到 LangChain 棄用

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

# Examples of a pretend task of creating antonyms.
few_shot_examples = [
    {"input": "happy", "output": "sad"},
    {"input": "tall", "output": "short"},
    {"input": "energetic", "output": "lethargic"},
    {"input": "sunny", "output": "gloomy"},
    {"input": "windy", "output": "calm"},
]

prompt_template = PromptTemplate(
    input_variables=["input", "output"],
    template="Input: {input}\nOutput: {output}",
)

# 跑 length based selector 必備套件
example_selector = LengthBasedExampleSelector(
    examples = few_shot_examples,
    example_prompt = prompt_template,
    max_length = 25,
)

few_shot_prompt = FewShotPromptTemplate(
    # We provide an ExampleSelector instead of examples.
    example_selector = example_selector,
    example_prompt = prompt_template,
    prefix="Give the antonym of every input",
    suffix="Input: {input}\nOutput:",
    input_variables=["input"],
)

chain = few_shot_prompt | llm
response = chain.invoke({"input": "principal"})
print(response.content)
