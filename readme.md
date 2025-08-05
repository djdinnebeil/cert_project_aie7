Step 1: Define the Core Idea 
1. Problem Statement (1 sentence):
When asked detailed questions about historical events, historians often must spend hours searching through fragmented and unstructured sources to provide accurate, reliable answers.

2. Why This Is a Problem (1–2 paragraphs):
Historians are frequently approached by members of the public—journalists, students, educators, or even local residents—with specific questions about events, organizations, or people from the past. These questions might seem straightforward—“What happened at Camp Amatol?”, “Who built the munitions factory in 1918?”, or “Was the Atlantic Loading Company connected to the war effort?”—but answering them accurately often requires extensive research.

Even when the historian has a general idea, they may need to verify names, dates, or relationships across a wide range of primary and secondary sources. These sources are often digitized without metadata, poorly organized, or span multiple formats like newspaper clippings, war department reports, and local records. To respond with confidence and precision, the historian might spend hours manually reviewing text, cross-referencing sources, and ensuring nothing is misrepresented. This slows the process of public engagement and increases the risk of oversight—particularly when quick, verifiable answers are expected in a digital-first world.

Step 2: Propose a Solution
## ✅ Task 2: Propose a Solution

### 1. Proposed Solution

To support historians in answering detailed questions from the public or verifying their own interpretations, I propose a document-grounded assistant that allows users to upload collections of primary and secondary sources—such as newspaper clippings, scanned memos, and war department reports—and then query those documents using natural language. The system will respond with fact-based answers that include citations or links to the exact documents and excerpts they came from.

A historian might upload a folder containing 30 files and ask:

> “When did construction begin at Camp Amatol?”

The assistant will retrieve relevant documents, extract the most pertinent chunks, and return an answer like:

> “Construction began in March 1918, according to the War Department report titled ‘Logistical Development Overview’.”

The goal is to significantly reduce the time required to verify information, while helping users trace the provenance of historical claims with transparency. The system will begin with naive retrieval and later integrate advanced retrieval components like reranking and agentic reasoning for more nuanced fact tracing.

---

### 2. Tool Stack

| Layer               | Tool                    | Rationale                                                                 |
|---------------------|-------------------------|---------------------------------------------------------------------------|
| **LLM**             | `gpt-4` (initially `gpt-3.5-turbo`) | Chosen for its strong reasoning and summarization capabilities; fallback to cheaper model for prototyping |
| **Embedding Model** | `text-embedding-3-small`| Cost-efficient and performant for initial semantic search                |
| **Orchestration**   | LangChain               | Modular integration of retriever, prompt templates, and agent logic      |
| **Vector Database** | FAISS (local, in-memory)| Lightweight, fast, easy to swap later for Qdrant or Chroma               |
| **Monitoring**      | LangSmith               | Provides tracing and visibility for prompt flows and agents              |
| **Evaluation**      | RAGAS                   | Quantifies faithfulness, relevance, and context quality for iterative improvement |
| **User Interface**  | Streamlit               | Quick to prototype, allows file uploads and natural language query interface |
| **(Optional) Serving** | FastAPI              | Enables turning the app into a backend service for long-term deployment  |

---

### 3. Agent Usage and Agentic Reasoning

Agents will be introduced in the advanced pipeline phase. Specifically, agents will:

- Route queries to the appropriate tools (e.g., extract text from a file or vector retrieval for local sources)
- Validate claims across multiple documents before answering
- Filter or re-rank results based on reasoning (e.g., preferring sources with date metadata or original authorship)
- Handle fallback behavior, such as when no relevant documents are found (invoke an external search agent or notify the user)

In essence, agentic reasoning will help structure multi-step reasoning tasks like:

> “Was Camp Amatol considered a strategic site by the War Department in 1918?”

This can’t be solved by retrieval alone—it benefits from chaining document lookups, entity matching, and synthesizing evidence from multiple sources.

## ✅ Task 3: Dealing with the Data

### 1. Data Sources and External APIs

The application will rely on a combination of **primary** and **secondary** historical sources uploaded by the user. These include:
- Newspaper clippings (scanned and OCR-processed)
- Official war department reports
- Institutional memos
- Secondary source summaries and books converted to text

Each file will be treated as a standalone document and converted into a vector store for semantic search.

In the advanced pipeline, the app will optionally use an **external search API**, such as [Tavily](https://tavily.com/) or [SerpAPI](https://serpapi.com/), for agent-based fallback behavior. This will allow the system to:
- Retrieve missing context that isn't present in the user’s uploaded corpus
- Verify facts or supply corroborating evidence from public sources

The agent will determine when to call this external API (e.g., when confidence is low or retrieval is empty).

---

### 2. Chunking Strategy

The default chunking strategy will use a **character-based approach** with:
- **Fixed-size chunks of 500 characters**
- **No overlap**, to avoid duplicating token cost
- A post-processor to attach **document-level metadata** to each chunk (e.g., filename or document type)

This strategy is chosen because:
- It keeps short newspaper articles or memos as 1–2 coherent chunks
- It aligns well with GPT-3.5/GPT-4 context limits
- It ensures minimal fragmentation for concise primary sources

In the advanced pipeline, semantic-aware chunking (e.g., `RecursiveCharacterTextSplitter`, or sentence-based segmentation) may be introduced.

---

### 3. Additional Data Requirements (Optional)

Each chunk must retain metadata that allows the system to:
- Trace it back to its source file
- Display citation information alongside the answer
- Optionally include tags like estimated date or document type (if provided by the user or inferred later)

To support this, the system will extract and store:
- `source`: filename or uploaded title
- `doc_type`: e.g., "newspaper", "official_report", "memo"
- (Optional) `date_estimate`: a best guess at the publication or origin date

This metadata will be used both for answer citation and for advanced re-ranking/filtering in later stages.
