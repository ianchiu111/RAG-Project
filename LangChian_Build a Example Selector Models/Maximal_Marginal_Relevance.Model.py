'''
Maximal Marginal Relevance (MMR) Example Selector Model
1. have a CUDA-supported GPU run "pip install faiss-gpu" on terminal
2. don't have a CUDA-supported GPU run "pip install faiss-cpu" on terminal
'''

from langchain_core.example_selectors import MaxMarginalRelevanceExampleSelector
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import FewShotPromptTemplate, PromptTemplate
import os
from langchain_openai import ChatOpenAI, OpenAIEmbeddings


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

example_selector = MaxMarginalRelevanceExampleSelector.from_examples(
    few_shot_examples,
    # The embedding class used to measure semantic similarity.
    OpenAIEmbeddings(),
    # The VectorStore class that is used to store the embeddings and do a similarity search over.
    FAISS,
    # The number of examples to produce.
    k=2,
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
