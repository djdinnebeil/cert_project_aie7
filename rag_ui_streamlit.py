import os
import streamlit as st
from dotenv import load_dotenv
from langchain_community.vectorstores import Qdrant
from langchain_community.embeddings import OpenAIEmbeddings
from qdrant_client import QdrantClient
from langchain_community.chat_models import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import LLMChainFilter
from langchain.chains import LLMChain
from langchain.retrievers.contextual_compression import ContextualCompressionRetriever
from langchain_cohere import CohereRerank

load_dotenv()

os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")
os.environ["COHERE_API_KEY"] = os.getenv("COHERE_API_KEY")
os.environ["CO_API_KEY"] = os.getenv("COHERE_API_KEY")

# Testing
testing = True


# --- Setup ---
COLLECTION_NAME = "historical_docs"
K = 15

# --- Prompt template ---
template = """
You are a helpful historical research assistant.

Use the following historical documents to answer the question as accurately and factually as possible.

{context}

Question: {question}
Answer:"""

prompt = PromptTemplate(
    template=template,
    input_variables=["context", "question"]
)

# --- Init components (only once) ---
@st.cache_resource
def load_chain():
    embeddings = OpenAIEmbeddings()
    client = QdrantClient(host="localhost", port=6333)
    vectorstore = Qdrant(
        client=client,
        collection_name=COLLECTION_NAME,
        embeddings=embeddings
    )
    naive_retriever = vectorstore.as_retriever(search_kwargs={"k": 15})

    compressor = CohereRerank(model="rerank-v3.5")
    compression_retriever = ContextualCompressionRetriever(
        base_compressor=compressor, base_retriever=naive_retriever, top_k=12
    )

    llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0)
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=compression_retriever,
        chain_type_kwargs={"prompt": prompt},
        return_source_documents=True
    )

    return qa_chain, naive_retriever, compression_retriever

qa_chain, naive_retriever, compression_retriever = load_chain()

# --- Streamlit UI ---
st.title("Historical Research Assistant")
question = st.text_input("Ask a question about your documents:")

if question:
    with st.spinner("Searching historical documents..."):
        # Optional diagnostics
        if testing:
            retrieved_docs = naive_retriever.get_relevant_documents(question)
            reranked_docs = compression_retriever.get_relevant_documents(question)

            for i, doc in enumerate(retrieved_docs):
                source = doc.metadata.get("source", "unknown")
                st.markdown(f"**Naive {i+1}** — `{source}`\n\n{doc.page_content[:300]}...")

            st.write("--------------------------------")
            for i, doc in enumerate(reranked_docs):
                source = doc.metadata.get("source", "unknown")
                st.markdown(f"**Reranked {i+1}** — `{source}`\n\n{doc.page_content}")

            st.info(f"Retrieved: {len(retrieved_docs)} chunks | Kept after compression: {len(reranked_docs)}")

        # Run the QA chain
        result = qa_chain(question)
        st.success("Answer ready!")

        st.subheader("📜 Answer")
        st.write(result["result"])

        st.subheader("📁 Sources consulted")
        # sources = set(doc.metadata.get("source", "unknown") for doc in result["source_documents"])
        sources = [doc.metadata.get("source", "unknown") for doc in result["source_documents"]]
        for source in sorted(sources):
            st.markdown(f"- `{source}`")
