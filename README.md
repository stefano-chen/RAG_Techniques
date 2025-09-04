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
3. Improve Query
    * Query tansformation techniques are used to enhance the retrieval process in RAG. There are 3 main techniques: **Query Rewritting**, **Step-back Prompting** and **Sub-query Decomposition**
        - Query Rewritting: Reformulates queries to be more specific and detailed.
        - Step-back Prompting: Generates broader queries for better context retrieval
        - Sub-query Decomposition: Breaks down complex queries into simple sub-queries.