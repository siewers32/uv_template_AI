from fastapi import FastAPI
from sqlmodel import Session, select
from database import engine, init_db
from models import Document, QueryRequest, QueryResponse
from core.llm import get_embedding, generate_answer

app = FastAPI()

@app.on_event("startup")
def on_startup():
    init_db()

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