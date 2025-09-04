from utils.preprocessing import load_pdf, split_pdf
from utils.models import get_llm
from utils.vectorstore import create_FAISS_store, create_FAISS_retriever
from pathlib import Path
from langchain_core.prompts import PromptTemplate
from dotenv import load_dotenv

load_dotenv(".env")

# Constant declaration
DATA_PATH = Path("./data")
QUERY = "What is the main cause of climate change?"

PROMPT_TEMPLATE = "HUMAN: You are an assistant for question-answering tasks. " \
"Use the following pieces of retrieved context to answer the question. If you don't know the answer, just say that you don't know. " \
"Use three sentences maximum and keep the answer concise. \n" \
"Question: {question} \n" \
"Context: {context} \n" \
"Answer:"


if __name__ == "__main__":
    pdf_list = load_pdf(path_to_dir=DATA_PATH)
    chunks = split_pdf(documents=pdf_list, chunk_size=1000, chunk_overlap=200)
    vector_store = create_FAISS_store(documents=chunks)
    retriever = create_FAISS_retriever(vector_store=vector_store, k=3)
    relevant_chunks = retriever.invoke(input=QUERY)
    context = "\n\n".join([chunk.page_content for chunk in relevant_chunks])
    llm = get_llm(full_model_name="gemini-2.5-flash-lite")
    prompt_template = PromptTemplate.from_template(PROMPT_TEMPLATE)
    prompt = prompt_template.invoke(input={"question": QUERY, "context": context})
    response = llm.invoke(input=prompt)
    print(prompt.to_string())
    print(response.content)