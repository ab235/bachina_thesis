# Student Handbook RAG System

A Retrieval-Augmented Generation (RAG) system for question-answering over student handbooks and other text documents.

## What is RAG?

RAG combines semantic search with AI text generation:
1. **Retrieval**: Find relevant text chunks from your documents using vector similarity
2. **Augmented Generation**: Use those chunks as context for GPT to generate accurate answers

This prevents AI hallucinations by grounding answers in your actual documents.

## Features

- ✅ **Semantic Search**: Find relevant information by meaning, not just keywords
- ✅ **pgvector Integration**: Fast vector similarity search in PostgreSQL
- ✅ **Overlapping Chunks**: Prevents important context from being split
- ✅ **Citation Support**: Tracks which chunks answers come from
- ✅ **Batch Processing**: Efficient embedding generation for large documents
- ✅ **Retry Logic**: Handles transient API failures gracefully

## Prerequisites

- **Python 3.8+**
- **PostgreSQL 12+** with pgvector extension
- **OpenAI API Key** (for embeddings and chat)

## Installation

### 1. Install PostgreSQL and pgvector

**On Ubuntu/Debian:**
```bash
sudo apt-get install postgresql postgresql-contrib
```

Then install pgvector extension:
```bash
git clone https://github.com/pgvector/pgvector.git
cd pgvector
make
sudo make install
```

**On macOS:**
```bash
brew install postgresql pgvector
```

### 2. Create Database

```bash
# Start PostgreSQL
sudo service postgresql start  # Linux
brew services start postgresql  # macOS

# Create database
createdb handbook_rag
```

Enable pgvector extension:
```sql
psql handbook_rag
CREATE EXTENSION IF NOT EXISTS vector;
\q
```

### 3. Install Python Dependencies

```bash
pip install -r requirements.txt
```

### 4. Configure Environment Variables

Create a `.env` file in the project root:

```env
# OpenAI Configuration
OPENAI_API_KEY=your_openai_api_key_here
OPENAI_EMBED_MODEL=text-embedding-3-large
OPENAI_CHAT_MODEL=gpt-4o-mini

# PostgreSQL Configuration
PGHOST=localhost
PGPORT=5432
PGDATABASE=handbook_rag
PGUSER=postgres
PGPASSWORD=your_postgres_password

# RAG Parameters (optional - these are the defaults)
CHUNK_SIZE=1200
CHUNK_OVERLAP=200
TOP_K=6
```

**Get your OpenAI API key at**: https://platform.openai.com/api-keys

## Usage

### Step 1: Ingest a Document

Convert your text document into searchable embeddings:

```bash
python ingest.py handbook.txt --doc-title "Student Handbook 2024"
```

This will:
1. Read your text file
2. Split it into ~1200 character chunks (with 200 char overlap)
3. Generate embeddings for each chunk using OpenAI
4. Store everything in the database

**Note**: This step costs money (OpenAI API usage) but only needs to be done once per document.

### Step 2: Ask Questions

Query your document with natural language questions:

```bash
python ask.py "What is the refund policy?"
python ask.py "How do I drop a class?"
python ask.py "What are the graduation requirements?"
```

The system will:
1. Convert your question to an embedding
2. Find the 6 most similar chunks from the database
3. Use GPT to generate an answer based on those chunks
4. Return the answer with chunk citations
