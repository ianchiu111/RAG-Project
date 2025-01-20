# 較差的 few-shot Prompting Model
## Try to Pass the examples and formatter to FewShotPromptTemplate
import os
from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate, FewShotPromptTemplate
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
)

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
# Different way to print out 
## Print out like a prompt and response
prompt = prompt_template_2.invoke(examples[0])
response = llm.invoke(prompt)
print(response.content)
## Print out like QA storytelling 
print(prompt_template_2.invoke(examples[0]).to_string())



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

## FewShotPromptTemplate have to use PromptTemplate Package
print(prompt.invoke({"input": "Who was the father of Mary Ball Washington?"}).to_string())

