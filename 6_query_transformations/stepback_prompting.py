from langchain.chat_models import init_chat_model
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv

load_dotenv("./.env")

# Constant Definition
STEPBACK_PROMPT = """You are an AI assistant tasked with generating broader, more general queries to improve context retrieval in a RAG system.
Given the original query, generate a step-back query that is more general and can help retrieve relevant background information.
Only respond with the step-back query.
Original query: {original_query}
"""

llm = init_chat_model(model="gemini-2.5-flash-lite", model_provider="google_genai")

rewrite_prompt = PromptTemplate.from_template(template=STEPBACK_PROMPT)

rewrite_chain = rewrite_prompt | llm

response = rewrite_chain.invoke({"original_query": "What are the impacts of climate change on the environment?"})

print(response.content)