# Dynamic Few-Shot Examples in Chat Models
## Replace the (multiple types of) examples passed into FewShotChatMessagePromptTemplate with an example_selector
## Before install Chroma, make sure C++ environment is established
### Install with <https://visualstudio.microsoft.com/zh-hant/visual-cpp-build-tools/>

import os
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_chroma import Chroma
from langchain_core.example_selectors import SemanticSimilarityExampleSelector
from langchain_openai import OpenAIEmbeddings
from langchain_core.prompts import FewShotChatMessagePromptTemplate


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

few_shot_examples = [
    # Examples type 1 --- Math Question
    {"input": "2 🦜 2", "output": "4"},
    {"input": "2 🦜 3", "output": "5"},
    {"input": "2 🦜 4", "output": "6"},

    # Examples type 2 --- QA Example
    {"input": "What did the cow say to the moon?", "output": "nothing at all"},
    {"input": "How many legs does the horses have?", "output": "horses have 4 legs"},
    
    # Examples type 3 --- Poetry Writing
    {
        "input": "Write me a poem about the moon",
        "output": "One for the moon, and one for me, who are we to talk about the moon?",
    },
]
# 將文字向量化
## 將範例用空格串成字串，供 Embedding 使用
to_vectorize = [" ".join(example.values()) for example in few_shot_examples]
## 建立一個 Embedding Object(物件) → 把文字轉成向量
embeddings = OpenAIEmbeddings()
## 利用 "Chroma.from_texts" 建立向量資料庫
vectorstore = Chroma.from_texts(to_vectorize, embeddings, metadatas=few_shot_examples)

# 利用 "SemanticSimilarityExampleSelector"  找出與 Prompt 相似度最高的 few-shot 範例
## 不會直接產出 final_prompt, 呼叫 llm 
example_selector = SemanticSimilarityExampleSelector(
    vectorstore = vectorstore,
    k = 2, ## 對 Prompt，只選出前 2 個最相似的範例
)
## check which examples will pick with user inputs
''' Generation Part
selected = example_selector.select_examples({"input": "horse"})
print(selected)
'''

# Define Few-Shot Prompt Template
## Must to combine ChatPromptTemplate or chain to make final prompt
few_shot_prompt = FewShotChatMessagePromptTemplate(
    example_prompt = prompt_template,
    example_selector = example_selector,
    # 若使用 static few-shot model 則將 example_selector 改為 examples
    # examples = few_shot_examples, 
)

# Combine to make final prompt
final_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "You are genious, you can answer everything."),
        few_shot_prompt,
        ("user", "{input}"),
    ]
)

# chain = final_prompt | llm_2 是一種 pipeline 的概念，將 final_prompt 的輸出直接接到 llm_2
# final_prompt 負責：把 system 訊息、few-shot examples、user 的 input 一起組合成最終完整的 Chat 訊息列表。
chain = final_prompt | llm

# Different Results between normal text response and few-shot 
response = chain.invoke({"input": "Kangaroo"})
print(response.content)

''' 錯誤 Generation Example
此方法模型無法理解 few-shot examples
response = llm.invoke({"input": "Kangaroo"})
print(response.content)
'''