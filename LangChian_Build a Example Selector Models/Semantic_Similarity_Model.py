'''
Semantic Similarity Example Selector Model
適用情景：
1. To find the examples with the embeddings that have the 
greatest cosine similarity with the inputs.
'''

from langchain_core.example_selectors import SemanticSimilarityExampleSelector
from langchain_core.prompts import FewShotPromptTemplate, PromptTemplate
import os
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_chroma import Chroma
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
example_selector = SemanticSimilarityExampleSelector.from_examples(
    # The list of examples available to select from.
    few_shot_examples,
    # The embedding class used to produce embeddings which are used to measure semantic similarity.
    OpenAIEmbeddings(),
    # The VectorStore class that is used to store the embeddings and do a similarity search over.
    Chroma,
    # The number of examples to produce.
    k = 2,
)

few_shot_prompt = FewShotPromptTemplate(
    # We provide an ExampleSelector instead of examples.
    example_selector = example_selector,
    example_prompt = prompt_template,
    prefix="Give the antonym of every input",
    suffix="Input: {input}\n Output:",
    input_variables=["input"],
)

chain = few_shot_prompt | llm
response = chain.invoke({"input": "zookeeper"})
print(response.content)
