from utils.preprocessing import load_csv, split_csv
from utils.vectorstore import create_FAISS, get_FAISS_retriever, add_documents
from utils.models import fetch_embedding_model, fetch_llm
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains.retrieval import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from pathlib import Path
from dotenv import load_dotenv

load_dotenv("./.env")

# constants declaration
CSV_PATH = Path("./data/customers-100.csv")
SYSTEM_PROMPT = (
    "You are an assistant for question-answering tasks. "
    "Use the following pieces of retrieved context to answer"
    "the question. If you don't know the aswer, say that you "
    "don't know. Use three sentences maximum and keep the answer concise."
    "\n\n"
    "{context}"
)


# entrypoint
if __name__ == "__main__":
    docs = load_csv(path_to_csv=CSV_PATH)
    chunks = split_csv(documents=docs)
    vector_store = create_FAISS(embedding_model=fetch_embedding_model(model_name="sentence-transformers/all-mpnet-base-v2"))
    add_documents(faiss_vector_store=vector_store, documents=chunks)
    retriever = get_FAISS_retriever(faiss_vector_store=vector_store, k=5)
    prompt = ChatPromptTemplate.from_messages([
        ("system", SYSTEM_PROMPT),
        ("human", "{input}")
    ])

    # Create the question-answer chain
    question_answer_chain = create_stuff_documents_chain(llm=fetch_llm(model_name="gemini-2.5-flash-lite"), prompt=prompt)
    rag_chain = create_retrieval_chain(retriever=retriever, combine_docs_chain=question_answer_chain)


    # query the rag system
    answer = rag_chain.invoke(input={"input": "which company does sheryl Baxter work for?"})
    print(answer)