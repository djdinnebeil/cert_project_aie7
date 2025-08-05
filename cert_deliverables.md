## Task 1: Define the Core Idea 
1. Problem Statement (1 sentence):
When asked detailed questions about historical events, historians often spend significant time searching across fragmented sources to answer specific historical questions accurately.

2. Why This Is a Problem (1–2 paragraphs):
Historians are often asked targeted questions that require precise, well-sourced answers. A seemingly simple question—like “What was the exact date construction began at Amatol, NJ?”—may require digging through historical records, even if the historian is familiar with the topic.

While the public often expects quick, specific answers, historians must take time to verify names, dates, and relationships. Even well-known events may need to be rechecked to ensure accuracy, especially when working with fragmented or unstructured source material.

## Task 2: Propose a Solution

### 1. Proposed Solution
To help historians efficiently answer detailed public inquiries or verify their own interpretations, this project develops a document-grounded assistant that enables users to upload primary and secondary sources, such as newspaper clippings, scanned memos, and other documentation. The user can then query the system to receive a response with accurate, context-aware answers and include citations or document excerpts to support each claim. The goal is to significantly reduce the time required to verify information, which the assistant does by providing transparent fact-checking allowing historians to respond more quickly and confidently. 

### 2. Tool Stack

| Layer               | Tool                    | Rationale                                                                 |
|---------------------|-------------------------|---------------------------------------------------------------------------|
| **LLM**             | gpt-4 (initially gpt-3.5-turbo) | Offers strong reasoning and summarization; gpt-3.5 is cost-effective for early-stage testing |
| **Embedding Model** | text-embedding-3-small| Strikes a balance between cost and semantic search performance             |
| **Orchestration**   | LangChain               | Modular integration of retriever, prompt templates, and agent logic      |
| **Vector Database** | Qdrant                  | Scalable and production-ready; supports filtering and metadata-rich document retrieval|
| **Monitoring**      | LangSmith               | Provides tracing and visibility for prompt flows and agents              |
| **Evaluation**      | RAGAS                   | Supports structured evaluation of retrieval quality and answer faithfulness |
| **User Interface**  | Streamlit               | Ideal for prototyping with support for file upload and natural language inputs |
| **Serving** | Streamlit's built-in server | Streamlit serves both the frontend and backend for rapid prototyping and demo deployment |


### 3. Agent Usage and Agentic Reasoning

Agents in this application will coordinate tool use and apply reasoning to enhance the accuracy and relevance of answers. Specifically, agents will:
- Route queries to appropriate tools, such as extracting text from uploaded files or retrieving relevant chunks from the vector store.
- Validate claims across multiple documents before responding, improving factual accuracy and reducing hallucinations.
- Filter or re-rank retrieved results using reasoning strategies—for example, prioritizing documents with date metadata or verified authorship.
- Handle fallback scenarios, such as when no relevant documents are found, by invoking an external web search agent or informing the user with a reasoned summary.

## ✅ Task 3: Dealing with the Data

### 1. Data Sources and External APIs

The application will operate primarily on user-uploaded historical documents. These include:
- OCR-processed newspaper clippings
- Official government documentation
- Miscellaneous primary sources
- Various secondary sources

Each file will be ingested as a standalone document and chunked for semantic search using a vector store.

The user can also request that the system perform a quick search, which can be useful for cross-verification or if there are not any relevant documents.

### 2. Chunking Strategy
The application will use a character-based chunking strategy by default, configured as follows:
- Fixed chunk size of 550 characters
- Overlap size of 50 characters
- Post-processing of document-level metadata, such as filename or source type

This approach was selected because it keeps short documents, such as newspaper clippings or memos, intact within 1-2 chunks. This helps to avoid unnecessary fragmentation for concise, information-dense sources.

### 3. Additional Data Requirements (Optional)

Each chunk must retain metadata to allow the system to display citation information alongside the answer. 

## ✅ Task 4: Build a Quick End-to-End Agentic Rag Prototype

The system is a document-grounded historical research assistant that enables users to upload primary and secondary sources, ask natural language questions, and receive evidence-backed answers.

The GitHub repo: https://github.com/djdinnebeil/cert_project_aie7

### Task 5: Creating a Golden Test Data Set

To evaluate the initial pipeline, I generated a small **synthetic test set** of historical questions based on the uploaded documents and ran them through the **naive retrieval pipeline** using the `RAGAS` framework. The table below summarizes the results across key evaluation metrics:

#### 📊 RAGAS Evaluation Results

| Pipeline | Total Runtime | Avg Latency | Total Tokens | Total Cost | Avg Tokens/Query | ContextRecall | Faithfulness | AnswerRelevancy | ContextEntityRecall |
|----------|----------------|--------------|----------------|-------------|--------------------|----------------|---------------|------------------|-----------------------|
| **Naive** | 38.57 sec     | 3.51 sec     | 29,161         | $0.003235   | 2,651.00           | 0.9848         | 0.9109        | 0.8404           | 0.3707                |

---

### Conclusion

The naive pipeline performs reasonably well in terms of **context recall**, **faithfulness**, and **answer relevancy**, showing that relevant context is being retrieved and grounded answers are being generated. However, **context entity recall** is significantly lower, indicating that while relevant chunks are retrieved, they may lack the necessary granularity or specificity (e.g., named entities or dates) for certain questions.

This baseline highlights a clear opportunity for improvement in retrieval precision, particularly by integrating advanced techniques such as **reranking**, **multi-query**, or **semantic-aware chunking** in the next stage of development.

### Task 6: The Benefits of Advanced Retrieval

To improve retrieval quality and response grounding, I implemented and tested a set of advanced retrieval techniques, building on the baseline established in Task 5.

---

#### 1. Retrieval Techniques Explored

- **Naive** – This retriever provides a baseline for comparison and performance benchmarking. 
- **BM25** – This retriever is lexically, keyword-based that is helpful for exact matches on names, dates, or uncommon terms that embeddings might miss.
- **Multi-Query Retrieval** – This retriever reformulates the original question into multiple paraphrased variants to improve recall by capturing semantically diverse matches.
- **Contextual Compression** – This retriever uses reranking via Cohere Rerank on the top-k naive results to improve precision by filtering out low-relevance or noisy chunks.
- **Semantic Chunking** – This retriever uses sentence-aware chunking to preserve coherence within chunks reducing fragmentation and improving retrieval quality for complex questions.
- **Ensemble** – This retriever merges results from multiple retrievers (e.g., BM25 + dense) to balance semantic and lexical relevance while leveraging document trustworthiness.
- **Ensemble + Compression** – This retriever adds contextual compression to maximize recall and precision while minimizing irrelevant or redundant context.

#### 2. Testing and Impact

These advanced retrieval pipelines were tested and evaluated on the following RAGAS and runtime metrics:

- **Total Runtime** – Total wall-clock time (in seconds) taken to process all queries for the given pipeline.

- **Avg Latency** – Average time (in seconds) taken to respond to a single query, including retrieval and generation.

- **Total Tokens** – Total number of tokens used across all queries, including both prompt and completion tokens.

- **Total Cost** – Approximate cost (in USD) based on token usage and model pricing.

- **Avg Tokens/Query** – Average number of tokens consumed per query (useful for evaluating efficiency and verbosity).

- **ContextRecall** – Measures how completely the retrieved context captures the ground truth (ideal: 1.0).

- **Faithfulness** – Measures how factually grounded the generated answer is in the retrieved context (ideal: 1.0).

- **AnswerRelevancy** – Assesses how well the generated answer addresses the original user question (ideal: 1.0).

- **ContextEntityRecall** – Evaluates whether the key entities needed to answer the question are present in the retrieved context (often the most difficult metric to optimize).

**Metrics not used**:
- **FactualCorrectness** was excluded because the answers are already grounded in retrieved context, making **faithfulness** a more direct and interpretable proxy for factual alignment.

- **NoiseSensitivity** was not used because the focus was on evaluating core retrieval and generation quality, not robustness to adversarial or noisy input.


### Task 7: Assessing Performance

To assess the impact of advanced retrieval strategies, I evaluated eight different retrieval pipelines using the `RAGAS` framework, measuring key metrics such as context recall, faithfulness, answer relevancy, and context entity recall. The results are shown below:

#### 📊 RAGAS Evaluation Results

| Pipeline                     | Total Runtime | Avg Latency | Total Tokens | Total Cost | Avg Tokens/Query | ContextRecall | Faithfulness | AnswerRelevancy | ContextEntityRecall |
|-----------------------------|---------------|-------------|---------------|-------------|------------------|----------------|---------------|------------------|-----------------------|
| **Naive**                   | 🟡 38.57       | 🟡 3.51      | 🟡 29,161      | 🟡 0.003235  | 🟡 2,651.00        | 🟡 0.9848      | 🟡 0.9109     | 🟡 0.8404         | 🟡 0.3707              |
| **BM25**                    | 🟢 23.96       | 🟢 2.18      | 🟢 8,044       | 🟢 0.001321  | 🟢 731.27          | 🔴 0.9394      | 🟡 0.9318     | 🔴 0.7560         | 🔴 0.2320              |
| **Multi-Query**             | 🟡 49.15       | 🟡 4.47      | 🟡 44,605      | 🟡 0.005378  | 🟡 4,055.00        | 🟢 1.0000      | 🟡 0.9343     | 🟡 0.8293         | 🟡 0.3242              |
| **Contextual Compression (k=7)** | 🟢 27.01       | 🟢 2.45      | 🟢 8,046       | 🟢 0.001321  | 🟢 731.45          | 🟡 0.9697      | 🔴 0.8788     | 🔴 0.7561         | 🟢 0.3677              |
| **Contextual Compression (k=12)**| 🟢 28.79       | 🟢 2.62      | 🟢 8,023       | 🟢 0.001312  | 🟢 729.45          | 🟢 1.0000      | 🔴 0.8874     | 🟢 0.8484         | 🟢 0.4032              |
| **Semantic**                | 🟡 30.00       | 🟡 2.73      | 🔴 53,330      | 🔴 0.006005  | 🔴 4,848.18        | 🔴 0.9091      | 🟢 0.9545     | 🟡 0.8457         | 🔴 0.2980              |
| **Ensemble**                | 🔴 78.74       | 🔴 7.16      | 🔴 97,422      | 🔴 0.010780  | 🔴 8,856.55        | 🟢 1.0000      | 🟢 0.9740     | 🟢 0.9333         | 🟡 0.3213              |
| **Ensemble + Compression**  | 🔴 68.46       | 🔴 6.22      | 🟡 20,061      | 🟡 0.002938  | 🟡 1,823.73        | 🟡 0.9848      | 🟡 0.9221     | 🔴 0.7592         | 🟡 0.3523              |

---

### Reflections

From these results, there are two pipelines that stand out: contextual compression (k=12) and ensemble (without compression).

The `Contextual Compression (k=12)` pipeline offers the best overall balance between performance and efficiency. It achieved perfect context recall and the highest context entity recall, while maintainig strong faithfulness and anaswer relevancy. Additionally, it kept the token usage and cost low, making it ideal for scaling or deployment under resource constraints. 

The `Ensemble` pipeline produced the highest scores across key metrics—achieving perfect context recall, excellent faithfulness (0.9740), and the best answer relevancy (0.9333) among all pipelines tested. However, this came at a steep cost: it recorded the worst total runtime (78.74 sec), highest average latency (7.16 sec), and highest total token usage (97,422) and cost ($0.01078).

### Retrieval Pipeline Selected

The contextual compression (k=12) pipeline was selected because of an optimal balance of efficiency and performance. Compared to the initial naive pipeline, there was slight improvement with answer relevancy, context recall, and entity-level precision. There were also dramatic reductions in token usage, cost, runtime, and latency. While there was a slight drop in faithfulness, it was within acceptable bounds given the gains in efficiency and performance.

### 🔄 Delta: Contextual Compression (k=12) vs Naive Pipeline

This table shows the difference in performance between the Contextual Compression (k=12) pipeline and the Naive pipeline. Positive values indicate improved performance with k=12.

| **Metric**              | **Δ (k=12 - Naive)** |
|-------------------------|----------------------|
| Answer Relevancy        | **+0.0080**          |
| Context Entity Recall   | **+0.0325**          |
| Context Recall          | **+0.0152**          |
| Avg Latency             | **−0.8900**          |
| Avg Tokens/Query        | **−1921.5500**       |
| Total Cost              | **−0.001923**        |
| Total Runtime           | **−9.7800**          |
| Total Tokens            | **−21138.00**        |
| Faithfulness            | **−0.0235**          |

### Projected Improvements

Because the `Ensemble` pipeline achieved the highest scores in context recall, faithfulness, and answer relevancy, it represents a strong candidate for a high-accuracy retrieval mode. While it is not suitable as the default due to its high latency, token usage, and cost, it clearly demonstrates the upper bound of retrieval quality achievable with more aggressive strategies. In future iterations, the application can offer this pipeline as an optional mode that users can enable when dealing with complex or high-stakes historical queries, where precision and completeness are more important than speed or efficiency.
