from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser


load_dotenv("./.env")

loader = PyPDFLoader(file_path="./data/Understanding_Climate_Change.pdf")
docs = loader.load()

text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
chunks = text_splitter.split_documents(docs)

embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")
llm = init_chat_model(model="gemini-2.5-flash-lite", model_provider="google_genai")

vector_store = FAISS.from_documents(chunks, embedding=embedding_model)

retriever = vector_store.as_retriever(search_kwargs={"k":5})

summary_prompt_template = """
You are an assistant for summarization tasks. Summarize the following piece of context.
Context: <context>{context}</context>
"""

answer_prompt_template = """
You are an assistant for question-answering tasks.
Use the following pieces of retrieved context to answer the question. If you don't know the answer, just say that you don't know.
Use three sentences maximum and keep the answer concise.
"Question: {question}
"Context: {context}
"""

summary_prompt = PromptTemplate.from_template(summary_prompt_template)

answer_prompt = PromptTemplate.from_template(answer_prompt_template)

retrieved_docs = retriever.invoke(input="what is the main topic of the document?")

summary_chain = summary_prompt | llm

summarized_chunks = []

for doc in retrieved_docs:
    summarized_chunk = summary_chain.invoke(input={"context": doc.page_content})
    summarized_chunks.append(summarized_chunk.content)

context = "\n\n".join([chunks for chunks in summarized_chunks])

answer_chain = answer_prompt | llm | StrOutputParser()

response = answer_chain.invoke({"question": "what is the main topic of the document?", "context": context})

print(response)