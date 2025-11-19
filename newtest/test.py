"""
Agentic RAG with LangGraph + LangChain — fully commented, single-file example
-----------------------------------------------------------------------------
This script builds an *agentic RAG* system that can decide when to retrieve
context from a vector store and when to answer directly.

It follows (and compiles) the tutorial flow:
  1) Preprocess documents (load + split)
  2) Create a retriever tool from a vector store
  3) Node: generate_query_or_respond (LLM decides to retrieve or answer)
  4) Node: grade_documents (LLM checks relevance of retrieved docs)
  5) Node: rewrite_question (LLM improves the query if docs were irrelevant)
  6) Node: generate_answer (LLM answers using retrieved context)
  7) Assemble the graph with conditional routing
  8) Run the agentic RAG

Usage
-----
1) Install dependencies (see `requirements` below) and set your OpenAI API key:

   $ pip install -U langgraph "langchain[openai]" langchain-community langchain-text-splitters bs4
   $ export OPENAI_API_KEY=your_key_here

2) Run the script directly to see a demo:

   $ python agentic_rag.py

3) (Optional) Change the URL list or plug in your own documents.

Requirements (tested with recent versions)
------------------------------------------
langgraph
langchain
langchain-community
langchain-openai (comes via langchain[openai])
langchain-text-splitters
beautifulsoup4 (bs4)

Notes
-----
- The example uses OpenAI models via LangChain's `init_chat_model` helper.
- The vector store is an in-memory store for simplicity. Swap for FAISS/Chroma
  for persistence or larger corpora.
- Error handling is kept minimal for clarity.
"""

from __future__ import annotations

# Standard library
from typing import Literal
import os

# LangChain / LangGraph imports
from langchain_community.document_loaders import WebBaseLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

# In-memory vector store + embeddings
from langchain_core.vectorstores import InMemoryVectorStore  # simple demo store
from langchain_openai import OpenAIEmbeddings

# Tools: turn a retriever into a callable tool
# Depending on your installed version, the import path may vary.
# If this import fails, try: `from langchain.tools.retriever import create_retriever_tool`
from langchain.tools.retriever import create_retriever_tool

# Chat model bootstrapper (select provider/model by string)
from langchain.chat_models import init_chat_model

# Message helpers
from langchain_core.messages import convert_to_messages

# Pydantic schema for grading
from pydantic import BaseModel, Field

# LangGraph core
from langgraph.graph import MessagesState, StateGraph, START, END
from langgraph.prebuilt import ToolNode, tools_condition


# ---------------------------
# 0) Configuration
# ---------------------------
# Model names (adjust to what you have access to). "openai:gpt-4o" is used in the
# tutorial; any compatible chat model works.
CHAT_MODEL_NAME = "openai:gpt-4o"
EMBED_MODEL = OpenAIEmbeddings  # class, instantiated below

# Demo URLs from Lilian Weng's blog (as in the tutorial)
URLS = [
    "https://lilianweng.github.io/posts/2024-11-28-reward-hacking/",
    "https://lilianweng.github.io/posts/2024-07-07-hallucination/",
    "https://lilianweng.github.io/posts/2024-04-12-diffusion-video/",
]


# ---------------------------
# 1) Preprocess documents
# ---------------------------
def fetch_and_split_documents(urls: list[str]):
    """Fetch web pages and split them into small chunks for vector search.

    - WebBaseLoader fetches and parses the content (requires bs4)
    - RecursiveCharacterTextSplitter splits into overlapping chunks
    """
    # Load each URL into a list[Document]
    docs_nested = [WebBaseLoader(url).load() for url in urls]
    # Flatten into a single list of Document objects
    docs = [doc for sub in docs_nested for doc in sub]

    # Split into chunks for vectorization
    splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        chunk_size=100,  # keep small for the demo
        chunk_overlap=50,
    )
    splits = splitter.split_documents(docs)
    return splits


# ---------------------------
# 2) Create a retriever tool
# ---------------------------
def build_retriever_tool(doc_splits):
    """Index documents, create a retriever, and wrap it as a tool.

    Returns a LangChain Tool that can be invoked by the model.
    """
    # Build a tiny in-memory vector store (swap for FAISS/Chroma for persistence)
    vectorstore = InMemoryVectorStore.from_documents(
        documents=doc_splits,
        embedding=EMBED_MODEL(),
    )
    retriever = vectorstore.as_retriever()

    # Wrap retriever into a tool the LLM can call via tool-calls
    retriever_tool = create_retriever_tool(
        retriever,
        name="retrieve_blog_posts",
        description="Search and return information about Lilian Weng blog posts.",
    )
    return retriever_tool


# ---------------------------
# 3) Generate query or respond (node)
# ---------------------------
# Initialize a deterministic chat model to steer decisions
response_model = init_chat_model(CHAT_MODEL_NAME, temperature=0)

def generate_query_or_respond(state: MessagesState):
    """LLM node that decides: call the retriever tool or answer directly.

    - Receives the current graph `state` which contains a list of chat messages.
    - Binds the retriever tool so the model can decide to call it.
    - Returns a new message (either a direct reply or a tool-call request).
    """
    response = response_model.bind_tools([global_tools["retriever_tool"]]).invoke(
        state["messages"]
    )
    return {"messages": [response]}


# ---------------------------
# 4) Grade documents (node + schema)
# ---------------------------
GRADE_PROMPT = (
    "You are a grader assessing relevance of a retrieved document to a user question.\n"
    "Here is the retrieved document: \n\n {context} \n\n"
    "Here is the user question: {question} \n"
    "If the document contains keyword(s) or semantic meaning related to the user question,\n"
    "grade it as relevant.\n"
    "Give a binary score 'yes' or 'no' to indicate whether the document is relevant."
)

class GradeDocuments(BaseModel):
    """Structured output schema for grading relevance."""

    binary_score: str = Field(
        description="Relevance score: 'yes' if relevant, or 'no' if not relevant"
    )

# Use the same deterministic model for grading
grader_model = init_chat_model(CHAT_MODEL_NAME, temperature=0)

def grade_documents(state: MessagesState) -> Literal["generate_answer", "rewrite_question"]:
    """Conditional edge function: decide next node based on relevance.

    - Reads the user's original question and the tool result content from `state`.
    - Asks the LLM to output a structured GradeDocuments with yes/no.
    - Routes to `generate_answer` if relevant; otherwise to `rewrite_question`.
    """
    # First user message is the question; last message is the tool result content
    question = state["messages"][0].content
    context = state["messages"][-1].content

    prompt = GRADE_PROMPT.format(question=question, context=context)
    response = grader_model.with_structured_output(GradeDocuments).invoke(
        [{"role": "user", "content": prompt}]
    )
    score = (response.binary_score or "").strip().lower()

    return "generate_answer" if score == "yes" else "rewrite_question"


# ---------------------------
# 5) Rewrite question (node)
# ---------------------------
REWRITE_PROMPT = (
    "Look at the input and try to reason about the underlying semantic intent / meaning.\n"
    "Here is the initial question:"
    "\n ------- \n"
    "{question}"
    "\n ------- \n"
    "Formulate an improved question:"
)

def rewrite_question(state: MessagesState):
    """LLM node that rewrites the user's query for better retrieval."""
    messages = state["messages"]
    question = messages[0].content
    prompt = REWRITE_PROMPT.format(question=question)
    response = response_model.invoke([
        {"role": "user", "content": prompt}
    ])
    # Return a *user* message with the improved question so downstream nodes see it
    return {"messages": [{"role": "user", "content": response.content}]}


# ---------------------------
# 6) Generate answer (node)
# ---------------------------
GENERATE_PROMPT = (
    "You are an assistant for question-answering tasks. "
    "Use the following pieces of retrieved context to answer the question. "
    "If you don't know the answer, just say that you don't know. "
    "Use three sentences maximum and keep the answer concise.\n"
    "Question: {question} \n"
    "Context: {context}"
)

def generate_answer(state: MessagesState):
    """LLM node that synthesizes a concise answer from retrieved context."""
    question = state["messages"][0].content
    context = state["messages"][-1].content
    prompt = GENERATE_PROMPT.format(question=question, context=context)
    response = response_model.invoke([{"role": "user", "content": prompt}])
    return {"messages": [response]}


# ---------------------------
# 7) Build the graph
# ---------------------------
# We'll stash tools in a module-level dict so nodes can reference them
global_tools = {}

def build_graph(retriever_tool):
    """Assemble the LangGraph workflow with conditional routing."""
    # Make tools accessible from nodes
    global_tools["retriever_tool"] = retriever_tool

    workflow = StateGraph(MessagesState)

    # Register nodes by function or name
    workflow.add_node(generate_query_or_respond)
    workflow.add_node("retrieve", ToolNode([retriever_tool]))
    workflow.add_node(rewrite_question)
    workflow.add_node(generate_answer)

    # Entry point
    workflow.add_edge(START, "generate_query_or_respond")

    # Decide whether to call tools (retriever) or end with direct response
    workflow.add_conditional_edges(
        "generate_query_or_respond",
        tools_condition,  # inspects the last AI message for tool calls
        {
            "tools": "retrieve",  # if tools were requested → run ToolNode
            END: END,               # otherwise, we're done
        },
    )

    # After retrieval, grade relevance and route accordingly
    workflow.add_conditional_edges(
        "retrieve",
        grade_documents,  # returns "generate_answer" or "rewrite_question"
    )

    # Normal edges
    workflow.add_edge("generate_answer", END)
    workflow.add_edge("rewrite_question", "generate_query_or_respond")

    # Compile into an executable graph
    return workflow.compile()


# ---------------------------
# 8) Demo run helper
# ---------------------------
def demo_run(graph):
    """Stream a demo question through the graph and print node updates."""
    print("\n=== Running Agentic RAG Demo ===\n")
    user_question = "What does Lilian Weng say about types of reward hacking?"

    for chunk in graph.stream({"messages": [{"role": "user", "content": user_question}]}):
        for node, update in chunk.items():
            # Pretty print each node's last message
            print(f"Update from node: {node}")
            last_msg = update["messages"][-1]

            # If the message has a convenience pretty_print, use it
            if hasattr(last_msg, "pretty_print"):
                last_msg.pretty_print()
            else:
                # Fallback: print role + content
                role = getattr(last_msg, "role", "assistant")
                content = getattr(last_msg, "content", str(last_msg))
                print(f"[{role}] {content}")
            print("\n")


# ---------------------------
# Main entry
# ---------------------------
if __name__ == "__main__":
    # Ensure we have an API key set
    if not os.environ.get("OPENAI_API_KEY"):
        raise RuntimeError(
            "OPENAI_API_KEY is not set. Export it before running: export OPENAI_API_KEY=..."
        )

    # 1) Fetch + split
    doc_splits = fetch_and_split_documents(URLS)

    # 2) Retriever tool
    retriever_tool = build_retriever_tool(doc_splits)

    # 7) Build the graph
    graph = build_graph(retriever_tool)

    # 8) Demo run
    demo_run(graph)
