import streamlit as st
from langchain.vectorstores import Qdrant
from langchain.embeddings import OpenAIEmbeddings
from qdrant_client import QdrantClient
from langchain.chat_models import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA
import os
from dotenv import load_dotenv

load_dotenv()

os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")

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
    retriever = vectorstore.as_retriever(search_kwargs={"k": K})
    llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0)
    return RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        chain_type_kwargs={"prompt": prompt},
        return_source_documents=True
    )

qa_chain = load_chain()

# --- Streamlit UI ---
st.title("Historical Research Assistant")
question = st.text_input("Ask a question about your documents:")

if question:
    with st.spinner("Searching historical documents..."):
        result = qa_chain(question)
        st.success("Answer ready!")

        st.subheader("📜 Answer")
        st.write(result["result"])

        st.subheader("📁 Sources consulted")
        sources = set(doc.metadata.get("source", "unknown") for doc in result["source_documents"])
        for source in sorted(sources):
            st.markdown(f"- `{source}`")
