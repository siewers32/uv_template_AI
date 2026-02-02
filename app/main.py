from fastapi import FastAPI, Depends
from sqlmodel import Session, select
from database import engine, init_db
from models import Document, QueryRequest, QueryResponse
from core.llm import get_embedding, generate_answer
from dotenv import load_dotenv
from contextlib import asynccontextmanager

# Laad de .env file
load_dotenv()

@asynccontextmanager
async def lifespan(app: FastAPI):
    # --- Code hieronder wordt uitgevoerd bij het OPSTARTEN ---
    print("Database initialiseren...")
    await init_db()
    
    yield  # De applicatie draait nu
    
    # --- Code hieronder wordt uitgevoerd bij het AFSLUITEN ---
    print("Applicatie wordt afgesloten...")
    # Hier kun je bijv. database-pools of connecties met LLM-services netjes sluiten

# Geef de lifespan functie mee bij het aanmaken van de app
app = FastAPI(title="RAG API", lifespan=lifespan)

@app.post("/ask", response_model=QueryResponse)
async def ask(request: QueryRequest):
    # 1. Vraag omzetten naar vector
    query_vec = get_embedding(request.question)
    
    # 2. Vector search in Postgres
    with Session(engine) as session:
        statement = (
            select(Document)
            .order_by(Document.embedding.max_compare_neighbors(query_vec))
            .limit(3)
        )
        results = session.exec(statement).all()
    
    # 3. Context samenstellen
    context = "\n".join([r.content for r in results])
    
    # 4. Antwoord genereren
    answer = generate_answer(request.question, context)
    
    return {
        "answer": answer,
        "sources": [r.content for r in results]
    }