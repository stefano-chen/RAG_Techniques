from langchain_core.prompts import PromptTemplate
from langchain.chat_models import init_chat_model
from dotenv import load_dotenv

load_dotenv("./.env")

# constant definition
query = "what are the impacts of climate change on the environment?"

REWRITE_TEMPLATE = "You are an AI assistant tasked with reformulating user queries to improve retrieval in a RAG system.\n" \
"Given the original query, rewrite it to be more specific, detailed, and likely to retrieve relevant information.\n" \
"Respond only with the reformulated query.\n Original query: {original_query}\n"


rewrite_llm = init_chat_model(model="gemini-2.5-flash-lite", model_provider="google_genai")

rewrite_prompt = PromptTemplate.from_template(REWRITE_TEMPLATE)

rewrite_chain = rewrite_prompt | rewrite_llm

rewrited_query = rewrite_chain.invoke({"original_query": query})

print(rewrited_query.content)