from langchain_community.vectorstores import Qdrant
from langchain_openai.embeddings import OpenAIEmbeddings
from qdrant_client import QdrantClient
from langchain_community.chat_models import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA
import os

os.environ["OPENAI_API_KEY"] = input("Enter your OpenAI API Key:")

# --- Config ---
COLLECTION_NAME = "historical_docs"
K = 10

# --- Inputs ---
question = "When did construction begin at Amatol, NJ?"

# --- Connect to Qdrant ---
embedding_fn = OpenAIEmbeddings()
client = QdrantClient(host="localhost", port=6333)

vectorstore = Qdrant(
    client=client,
    collection_name=COLLECTION_NAME,
    embeddings=embedding_fn
)

retriever = vectorstore.as_retriever(search_kwargs={"k": K})

# --- RAG Prompt Template ---
template = """
You are a helpful historical research assistant.

Use the following historical documents to answer the question as accurately and factually as possible.

{context}

Question: {question}
Answer:"""

rag_prompt = PromptTemplate(
    template=template,
    input_variables=["context", "question"]
)

# --- LLM ---
llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0)

# --- Build RAG Chain ---
rag_chain = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",  # Use a simple RAG pipeline
    retriever=retriever,
    chain_type_kwargs={"prompt": rag_prompt}
)

# --- Run ---
answer = rag_chain.run(question)
print(f"\n✅ Final Answer:\n{answer}")
