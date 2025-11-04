import bs4
from langchain.hub import pull
from langchain_community.document_loaders import WebBaseLoader
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langgraph.graph import START, StateGraph
from typing_extensions import List, TypedDict

from langchain.chat_models import init_chat_model

llm = init_chat_model("gpt-4o-mini", model_provider="openai")
from langchain_openai import OpenAIEmbeddings

embeddings = OpenAIEmbeddings(model="text-embedding-3-large")

from langchain_postgres import PGVector

vector_store = PGVector(
    embeddings=embeddings,
    collection_name="my_docs",
    connection="postgresql+psycopg://...",
)

from pathlib import Path
import tempfile
import requests

from langchain_community.document_loaders import UnstructuredPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

# --- Option A: local PDF path ---
pdf_path = "rpi-student-handbook.pdf"  # point to your local PDF

# --- Option B: fetch a PDF from a URL (saves to a temp file) ---
# url = "https://arxiv.org/pdf/1706.03762.pdf"
# tmp = Path(tempfile.gettempdir()) / "download.pdf"
# tmp.write_bytes(requests.get(url, timeout=60).content)
# pdf_path = str(tmp)

# Unstructured loader:
# - mode="elements" -> one Document per detected element (headings, paragraphs, tables)
# - strategy: "fast" (text PDFs), "hi_res"/"ocr_only" (scanned)
loader = UnstructuredPDFLoader(
    file_path=pdf_path,
    mode="elements",
    strategy="fast",              # change to "hi_res" or "ocr_only" for scanned PDFs
    infer_table_structure=True,   # try to extract tables
)

docs = loader.load()
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200
)
all_splits = text_splitter.split_documents(docs)
# Index chunks
_ = vector_store.add_documents(documents=all_splits)

# Define prompt for question-answering
# N.B. for non-US LangSmith endpoints, you may need to specify
# api_url="https://api.smith.langchain.com" in hub.pull.
prompt = pull("rlm/rag-prompt")


# Define state for application
class State(TypedDict):
    question: str
    context: List[Document]
    answer: str


# Define application steps
def retrieve(state: State):
    retrieved_docs = vector_store.similarity_search(state["question"])
    return {"context": retrieved_docs}


def generate(state: State):
    docs_content = "\n\n".join(doc.page_content for doc in state["context"])
    messages = prompt.invoke({"question": state["question"], "context": docs_content})
    response = llm.invoke(messages)
    return {"answer": response.content}


# Compile application and test
graph_builder = StateGraph(State).add_sequence([retrieve, generate])
graph_builder.add_edge(START, "retrieve")
graph = graph_builder.compile()