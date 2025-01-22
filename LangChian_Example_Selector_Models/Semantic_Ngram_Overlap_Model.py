'''
Semantic Ngram Overlap Example Selector Model
1. need to "pip install nltk"
2. "N-gram" means to separate the documents into continuous strings into a sentence which length = n
3. threshold 的作用：
    a. 預設為 -1.0：表示不排除任何範例（只會依照分數排序，全部都給出）。
    b. 大於 1.0：表示會把所有範例都排除（因為 n-gram 分數通常介於 0 與 1 之間）。
    c. 等於 0.0：表示只有與輸入有任何 n-gram 重疊的範例才會被保留，沒有重疊就會被排除。
    d. 介於 0.0 和 1.0：表示只保留重疊度高於此數值的範例。
 
應用情境：
1. 語言建模
2. 文字預測
3. 機器翻譯
'''

from langchain_community.example_selectors import NGramOverlapExampleSelector
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
    {"input": "你好", "output": "Hello"},
    {"input": "這是我的家人", "output": "These are my family"},
    {"input": "我是碩士一年級的學生", "output": "I am a graduate student "},
]

prompt_template = PromptTemplate(
    input_variables=["input", "output"],
    template="Input: {input}\nOutput: {output}",
)

# 跑 length based selector 必備套件
example_selector = NGramOverlapExampleSelector(
    examples = few_shot_examples,
    example_prompt = prompt_template,
    threshold = -1.0,
    # Setup the N of N-gram
    n = 1
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
response = chain.invoke({"input": "這是我的狗"})
print(response.content)
