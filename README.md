# Local RAG Pipeline: F1 Document Q&A

This repository contains an end-to-end Retrieval-Augmented Generation (RAG) system built to parse, embed, and query text documents about Formula 1 racing teams.

## Features
- **Semantic Search**: Uses local HuggingFace embeddings (`all-MiniLM-L6-v2`) and ChromaDB to perform vector searches without hitting API rate limits during ingestion.
- **Maximal Marginal Relevance (MMR)**: Retrieves diverse context chunks to prevent redundant information.
- **High-Speed Inference**: Uses Groq's API and the `llama-3.1-8b-instant` model for lightning-fast and intelligent answers, bypassing the strict rate limits of traditional free-tier APIs.
- **Interactive Chat UI**: A sleek, conversational Streamlit frontend that preserves chat history.
- **Contradiction Checker**: A dedicated endpoint and UI tool to compare two different documents against a specific topic to identify conflicts.

## Tech Stack
* **Framework**: LangChain
* **Vector Database**: ChromaDB (Local)
* **Embeddings**: HuggingFace (`all-MiniLM-L6-v2` via `sentence-transformers`)
* **LLM**: Groq (`llama-3.1-8b-instant`)
* **Backend**: FastAPI & Uvicorn
* **Frontend**: Streamlit

## Setup Instructions

### 1. Install Dependencies
Ensure you have Python 3.9+ installed. Run the following command to install required packages:
```bash
pip install -r requirements.txt
```

### 2. Environment Variables
Create a `.env` file in the root of the project and add your Groq API key:
```env
GROQ_API_KEY=gsk_your_api_key_here
```
*(You can get a free API key at [console.groq.com/keys](https://console.groq.com/keys))*

### 3. Ingest Data
If you are running this for the first time or have added new `.txt` files to the directory, you need to build the Chroma vector database:
```bash
python ingest.py
```

### 4. Run the Application
You will need two terminal windows to run both the backend and frontend.

**Terminal 1 (Backend):**
```bash
uvicorn app:app --reload
```
**Terminal 2 (Frontend):**
```bash
streamlit run ui.py
```

Navigate to `http://localhost:8501` in your browser to interact with the RAG chat!

## My Learning Journey (A Beginner's Approach)
Coming into this project, I had to learn RAG architecture from scratch. I broke it down into chunking, embedding, vector search, and synthesis. 
During development, I learned how brittle external APIs can be when I hit severe rate limits and versioning conflicts with Gemini. To engineer around this, I pivoted to a hybrid approach: local open-source embeddings (HuggingFace) to build the vector DB for free, and Groq's high-speed cloud infrastructure for the final LLM synthesis step. This sprint taught me the fundamentals of vector databases and how to build resilient AI pipelines.