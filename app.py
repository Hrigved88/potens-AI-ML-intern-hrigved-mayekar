from fastapi import FastAPI
from pydantic import BaseModel
import os
from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_groq import ChatGroq

load_dotenv()

app = FastAPI()

# Match the local embedding model
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
vectorstore = Chroma(persist_directory="./chroma_db", embedding_function=embeddings)
llm = ChatGroq(model="llama-3.1-8b-instant")

class AskRequest(BaseModel):
    query: str

class ContradictRequest(BaseModel):
    doc1_name: str
    doc2_name: str
    topic: str

@app.post("/ask")
def ask(request: AskRequest):
    retriever = vectorstore.as_retriever(search_type="mmr", search_kwargs={"k": 3})
    docs = retriever.invoke(request.query)
    
    if not docs:
        return {"answer": "I do not know. No relevant context found.", "citations": []}

    context_text = ""
    citations = []
    for i, doc in enumerate(docs):
        source = doc.metadata.get('source', 'Unknown')
        snippet = doc.page_content[:150] + "..." 
        context_text += f"\n[Source: {source}]: {doc.page_content}\n"
        citations.append({"source": source, "snippet": snippet})

    prompt = f"""
    You are a strict technical assistant. Answer the user's question using ONLY the context provided below.
    If the context does not contain the answer, reply EXACTLY with: "I do not know based on the provided documents."
    Do not use outside knowledge. 
    CRITICAL: Answer in the exact same language that the user's Question is written in.

    Context:
    {context_text}

    Question: {request.query}
    """
    
    try:
        response = llm.invoke(prompt)
        final_answer = response.content
    except Exception as e:
        final_answer = f"Error generating answer: {str(e)}"
        
    return {"answer": final_answer, "citations": citations}

@app.post("/contradict")
def contradict(request: ContradictRequest):
    retriever = vectorstore.as_retriever(search_kwargs={"k": 4})
    docs1 = retriever.invoke(f"{request.doc1_name} {request.topic}")
    docs2 = retriever.invoke(f"{request.doc2_name} {request.topic}")
    
    context1 = "\n".join([d.page_content for d in docs1 if request.doc1_name in d.metadata.get('source', '')])
    context2 = "\n".join([d.page_content for d in docs2 if request.doc2_name in d.metadata.get('source', '')])

    prompt = f"""
    Analyze the following two document extracts regarding the topic: '{request.topic}'.
    Do they conflict or contradict each other? Provide a brief reasoning.
    
    Extracts from {request.doc1_name}: {context1}
    
    Extracts from {request.doc2_name}: {context2}
    """
    
    try:
        response = llm.invoke(prompt)
        analysis_result = response.content
    except Exception as e:
        analysis_result = f"Error generating analysis: {str(e)}"
        
    return {"analysis": analysis_result}