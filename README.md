# Techniques to improve RAG
This is a summary of the techniques presented [here](https://github.com/NirDiamant/RAG_Techniques)
1. Improve Raliability
    * Take the ```query``` as input
    * retrieve ```docs``` close to the query
    * check the ```retrieved document``` relevancy by feeding a ```LLM``` with the docs and query,\
    asking if the provided document are relevant to the query. Keep only the relevant ones
    * Generate the ```answer``` by feeding a ```LLM``` the ```relevant docs``` and the ```query```.
    * Using the ```relevant docs``` and the ```answer``` check if hallucinations are present \
    by asking a ```LLM``` if the ```generated answer``` is grounded / supported by the ```relevant docs```
2. Improve Chunking
    * high chunk size -> can introduce noise
    * low chunk size -> can lose informations
    * different chunking strategy : **slidding window**, **phase chunking**, **paragraph chunking**, **semantic chunking** 
    * using **Proposition chunking** to enhance chunks content, it consist of:
        - The input document is split into smaller piece (**chunks**). This ensure that each chunk is of manageable size for the ```LLM``` to process
        - Proposition are generated from **each** chunk using an ```LLM```. The output is structured as a list of factual, self-contained statements that can be understood without additional context.
        - A ```second LLM``` evaluates the quality of the propositions by scoring them on **accuracy**, **clarity**, **completeness**, and **conciseness**. Propositions that meet the required thresholds in all categories are **retained**.
        - Propositions that pass the quality check are ```embedded``` into a vector store.
    * **Semantic Chunking** aims to create more meaningful and context-aware text segments.
        - the data is readed
        - use LangChain's ```SemanticChunker``` with an embedding model. Three breakpoint types are available:
            - "percentile": splits at differences greater than the X percentile
            - "standard_deviation": Splits at differences greater than X standard deviations
            - "interquartile": uses the interquartile distance to determinne split points.
3. Improve Query
    * Query tansformation techniques are used to enhance the retrieval process in RAG. There are 3 main techniques: **Query Rewritting**, **Step-back Prompting** and **Sub-query Decomposition**
        - Query Rewritting: Reformulates queries to be more specific and detailed.
        - Step-back Prompting: Generates broader queries for better context retrieval
        - Sub-query Decomposition: Breaks down complex queries into simple sub-queries.
    * Hypothetical Document Embedding (**HyDE**) is a innovative approach that trasforms query questions into hypothetical documents containing the answer, aiming to bridge the gap between query and document distributions in vector space.
        - The documents are processed and split into chunks
        - the chunks are stored in a vector store using a embedding model
        - A LLM is used to generate a hypothetical document that answers the given query.
        - the generation is guided by a prompt that ensures the hypothetical document is detailed and matches the chunk size used in the vector store.
        - use the hypothetical document as the search query in the vector store
        - this retrieves the most similar documents to this hypothetical document
        - ```DOUBT```: this approach is based on the fact that the LLM used to generate the hypothetical knows the answer to the user's query. Then why don't use the LLM instead of building a RAG system? And if our working domain is very niche, then there is no guarantee that an LLM which knows the answer exists!
4. Context and Content Enrichment
    * **HyPE**(Hypothetical Prompt Embedding) is an enhancement to traditional RAG retrieval that precomputes hypothetical queries at the indexing stage, but inserting the chunk in their place. This transforms retrieval into a question-question matching task. This avoid the need for runtime synthetic answer generation, reducing inference-time computational overhead while improving retrieval alignment
        - Precomputed Questions: Instead of embedding document chunks, HyPE **generates multiple hypothetical queries per chunk** at indexing time.
        - Question-Question Matching: User queries are matched against stored hypothetical questions, leading to better retrieval alignment.
        - No Runtime Overhead: **Does not require LLM calls at query time, making retrieval fast and cheaper**
    * Contextual chunk headers (**CCH**) is a method of creating chunk headers that contain higher-level context (such as document-level or section-level context), and prepending those chunk headers to the chunks prior to embedding them. This givves the embeddings a much more accurate and complete representation of the content and meaning of the text.
        - use an LLM to generate a descriptive title for the document
        - you could also generate a concise document summary or section/sub-section title(s)
        - the text you embed for each chunk is simply the concatenation of the chunk header and the chunk text.
        - if you use a reranker during retrieval, you'll want to make sure you use this same concatenation there too.
        - Including the chunk headers when presenting the search results to the LLM is also beneficial as it gives the LLM more context, and makes it less likely that it misundestands the meaning of a chunk.
    * **Contextual Compression** is about compressing retrieved information while preseving query-relevant content.
        - retrieve chunks from the vector store
        - Use an ```LLM``` to compress or summerize retrieved chunks, preserving information relevant to the query.