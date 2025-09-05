from dotenv import load_dotenv
from langchain.prompts import PromptTemplate
from langchain_community.vectorstores import FAISS
from langchain_huggingface.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain_core.retrievers import BaseRetriever
from typing import Dict, Any
from langchain_core.documents import Document
from langchain.chat_models import init_chat_model
from pydantic import BaseModel, Field

load_dotenv("./.env")


# Define the query classifier class
class categories_options(BaseModel):
    category: str = Field(description="The category of the query, the options are: Factual, Analytical, Opinion, or Contextual",
                          example="Factual")

class QueryClassifier:
    def __init__(self) -> None:
        self.llm = init_chat_model(model="gemini-2.5-flash-lite", model_provider="google_genai")
        self.prompt = PromptTemplate(
            input_variables=["query"],
            template="Classify the following query into one of these categories: Factual, Analytical, Opinion, or Contextual.\nQuery: {query}\nCategory:"
        )
        self.chain = self.prompt | self.llm.with_structured_output(categories_options)

    def classify(self, query):
        print("classifying query")
        return self.chain.invoke(query).category # type: ignore
    
# Define the Base Retriever class, such that the complex ones will inherit from it
class BaseRetrievalStrategy:
    def __init__(self, texts) -> None:
        self.embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")
        text_splitter = CharacterTextSplitter(chunk_size=800, chunk_overlap=0)
        self.documents = text_splitter.create_documents(texts=texts)
        self.db = FAISS.from_documents(self.documents, self.embeddings)
        self.llm = init_chat_model(model="gemini-2.5-flash-lite", model_provider="google_genai")

    def retrieve(self, query, k=4):
        return self.db.similarity_search(query=query, k=k)
    
# Define Factual retriever strategy
class relevant_score(BaseModel):
    score: float = Field(description="The relevance score of the document to the query", example=8.0)

class FactualRetrievalStrategy(BaseRetrievalStrategy):
    def retrieve(self, query, k=4):
        print("retrieving factual")
        # Use LLM to enhance the query
        enhanced_query_prompt = PromptTemplate(
            input_variables=["query"],
            template="Enhance this factual query for better information retrieval: {query}"
        )
        query_chain = enhanced_query_prompt | self.llm
        enhanced_query = query_chain.invoke(query).content
        print(f"enhanced query: {enhanced_query}")

        # Retrieve documents using the enhanced query
        docs = self.db.similarity_search(enhanced_query, k=k*2) # type: ignore

        # Use LLM to rank the relevance of retrieved documents
        ranking_prompt = PromptTemplate(
            input_variables=["query", "doc"],
            template="On a scale of 1-10, how relevant is this document to the query: '{query}'?\nDocument: {doc}\nRelevance score:"
        )
        ranking_chain = ranking_prompt | self.llm.with_structured_output(relevant_score)

        ranked_docs = []
        print("ranking docs")
        for doc in docs:
            input_data = {"query": enhanced_query, "doc": doc.page_content}
            score = float(ranking_chain.invoke(input=input_data).score) # type: ignore
            ranked_docs.append((doc, score))

        # Sort by relevance score and return top k
        ranked_docs.sort(key=lambda x: x[1], reverse=True)
        return [doc for doc, _ in ranked_docs[:k]]
    
# Define Analytical retriever strategy
class SelectedIndices(BaseModel):
    indices: list[int] = Field(description="Indices of the selected documents", example=[0, 1, 2, 3])

class SubQueries(BaseModel):
    sub_queries: list[str] = Field(description="List of sub-queries for comprehensive analysis", 
                                   example=["What is the population of New York?", "What is the GDP of New York?"]

    )

class AnalyticalRetrievalStrategy(BaseRetrievalStrategy):
    def retrieve(self, query, k=4):
        print("retrieving analytical")
        # Use LLM to generate sub-queries for comprehensive analysis
        sub_queries_prompt = PromptTemplate(
            input_variables=["query", "k"],
            template="Generate {k} sub-questions for: {query}"
        )
        sub_queries_chain = sub_queries_prompt | self.llm.with_structured_output(SubQueries)

        input_data = {"query": query, "k": k}
        sub_queries = sub_queries_chain.invoke(input_data).sub_queries #type: ignore
        print(f"sub queries for comprehensive analysis: {sub_queries}")

        all_docs = []
        for sub_query in sub_queries:
            all_docs.extend(self.db.similarity_search(sub_query, k=2))

        # Use LLM to ensure diversity and relevance
        diversity_prompt = PromptTemplate(
            input_variables=["query", "docs", "k"],
            template="""Select the most diverse and relevant set of {k} documents for the query: '{query}'\nDocuments: {docs}\n
            Return only the indices of selected documents as a list of integers."""
        )
        diversity_chain = diversity_prompt | self.llm.with_structured_output(SelectedIndices)
        docs_text = "\n".join([f"{i}: {doc.page_content[:50]}..." for i, doc in enumerate(all_docs)])
        input_data = {"query": query, "docs": docs_text, "k": k}
        selected_indices_result = diversity_chain.invoke(input_data).indices #type: ignore
        print(f"selected diverse and relevant documents")

        return [all_docs[i] for i in selected_indices_result if i < len(all_docs)]
    

# Define Opinion retriever strategy
class OpinionRetrievalStrategy(BaseRetrievalStrategy):
    def retrieve(self, query, k=3):
        print("retrieving opinion")
        # Use LLM to identify potential viewpoints
        viewpoints_prompt = PromptTemplate(
            input_variables=["query", "k"],
            template="Identify {k} distinct viewpoints or perspectives on the topic: {query}"
        )
        viewpoints_chain = viewpoints_prompt | self.llm
        input_data = {"query": query, "k": k}
        viewpoints = viewpoints_chain.invoke(input_data).content.split("\n") #type: ignore
        print(f"viewpoints: {viewpoints}")

        all_docs = []
        for viewpoint in viewpoints:
            all_docs.extend(self.db.similarity_search(f"{query} {viewpoint}", k=2))

        # Use LLM to classify and select diverse opinions
        opinion_prompt = PromptTemplate(
            input_variables=["query", "docs", "k"],
            template="Classify these documents into distinct opinions on '{query}' and select the {k} most representative and diverse viewpoints:\nDocuments: {docs}\nSelected  indices:"
        )
        opinion_chain = opinion_prompt | self.llm.with_structured_output(SelectedIndices)

        docs_text = "\n".join([f"{i}: {doc.page_content[:100]}..." for i, doc in enumerate(all_docs)])
        input_data = {"query": query, "docs": docs_text, "k": k}
        selected_indices = opinion_chain.invoke(input_data).indices # type: ignore
        print(f"selected diverse and relevant documents")

        return [all_docs[int(i)] for i in selected_indices.split() if i.isdigit() and int(i) < len(all_docs)]

# Define Contextual retriever strategy
class ContextualRetrievalStrategy(BaseRetrievalStrategy):
    def retrieve(self, query, k=4, user_context=None):
        print("retrieving contextual")
        # Use LLM to incorporate user context into the query
        context_prompt = PromptTemplate(
            input_variables=["query", "context"],
            template="Given the user context: {context}\nReformulate the query to best address the user's needs: {query}"
        ) 
        context_chain = context_prompt | self.llm
        input_data = {"query": query, "context": user_context or "No specific context provided"}
        contextualized_query = context_chain.invoke(input_data).content
        print(f"contextualized query: {contextualized_query}")

        # Retrieve documents using the contextualized query
        docs = self.db.similarity_search(contextualized_query, k=k*2)

        # Use LLM to rank the relevance of retrieved documents considering the user context
        ranking_prompt = PromptTemplate(
            input_variables=["query", "context", "doc"],
            template="Given the query: '{query}' and user context: '{context}', rate the relevance of this document on a scale of 1-10:\nDocument: {doc}\nRelevance score:"
        )
        ranking_chain = ranking_prompt | self.llm.with_structured_output(relevant_score)
        print("ranking docs")

        ranked_docs = []
        for doc in docs:
            input_data = {"query": contextualized_query, "context": user_context or "No specific context provided", "doc": doc.page_content}
            score = float(ranking_chain.invoke(input_data).score)
            ranked_docs.append((doc, score))

        # Sort by relevance score and return top k
        ranked_docs.sort(key=lambda x: x[1], reverse=True)

        return [doc for doc, _ in ranked_docs[:k]]
    
# Define the Adaptive retriever class
class AdaptiveRetriever:
    def __init__(self, texts: list[str]):
        self.classifier = QueryClassifier()
        self.strategies = {
            "Factual" : FactualRetrievalStrategy(texts),
            "Analytical": AnalyticalRetrievalStrategy(texts),
            "Opinion": OpinionRetrievalStrategy(texts),
            "Contextual": ContextualRetrievalStrategy(texts)
        }
    
    def get_relevant_documents(self, query: str) -> list[Document]:
        category = self.classifier.classify(query=query)
        strategy = self.strategies[category]
        return strategy.retrieve(query)

# Define Additional retriever that inherits from langchain BaseRetriever
class PydanticAdaptiveRetriever(BaseRetriever):
    adaptive_retriever: AdaptiveRetriever = Field(exclude=True)

    class Config:
        arbitrary_types_allowed = True
    
    def get_relevant_documents(self, query: str) -> list[Document]:
        return self.adaptive_retriever.get_relevant_documents(query)
    
    async def aget_relevant_documents(self, query: str) -> list[Document]:
        return self.get_relevant_documents(query=query)
    
# Define the Adaptive RAG class
class AdaptiveRAG:
    def __init__(self, texts: list[str]):
        adaptive_retriever = AdaptiveRetriever(texts)
        self.retriever = PydanticAdaptiveRetriever(adaptive_retriever=adaptive_retriever)
        self.llm = init_chat_model(model="gemini-2.5-flash-lite", model_provider="google_genai")

        # Create a custom prompt
        prompt_template = """Use the following pieces of context to answer the question at the end.
        If you don't know the answer, just say that you don't know, don't try to make up an answer.
        
        {context}

        Question: {question}
        Answer:"""
        prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])

        # Create the LLM chain
        self.llm_chain = prompt | self.llm

    def answer(self, query: str) -> str:
        docs = self.retriever.get_relevant_documents(query)
        input_data = {"context": "\n".join([doc.page_content for doc in docs]), "question": query}
        return self.llm_chain.invoke(input_data)
    
# Usage
texts = [
    "The Earth is the third planet from the Sun and the only astronomical object know to harbor life."
]

rag_system = AdaptiveRAG(texts=texts)

factual_result = rag_system.answer("What is the distance between the Earth and the Sun?").content
print(f"Answer: {factual_result}")

analytical_result = rag_system.answer("How does the Earth's distance from the Sun affect it climate?").content
print(f"Answer: {analytical_result}")

opinion_result = rag_system.answer("What are the different theories about the origin of life on Earth?").content
print(f"Answer: {opinion_result}")

contextual_result = rag_system.answer("How does the Earth's position in the Solar System influence its habitability?").content
print(f"Answer: {contextual_result}")