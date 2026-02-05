from fastapi import Depends, FastAPI, Request, Form
from sqlalchemy.ext.asyncio import AsyncSession
from sqlmodel import select
from database import engine, get_session, init_db
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

# 2. De POST route: Om de ingevulde vraag te verwerken
def question_to_question_request(
        request: str = "vraag",
        ) -> QueryRequest:
    return QueryRequest(request=request)

@app.post("/")
async def verwerk_formulier(
     session: AsyncSession = Depends(get_session),
     request: QueryRequest = Depends(question_to_question_request), 
) -> QueryResponse:
    # 1. Vraag omzetten naar vector
    query_vec = await get_embedding(request.request) 
    
    # 2. Vector search met score (distance)
    async with AsyncSession(engine) as session:
        # We halen de afstand (distance) op om de relevantie te bepalen
        distance_col = Document.embedding.cosine_distance(query_vec).label("distance")
        statement = (
            select(Document, distance_col)
            .order_by(distance_col)
            .limit(5) # 10 is vaak veel, 5 is meestal zat voor context
        )
        results = await session.execute(statement)
        rows = results.all() # Geeft tuples van (Document, distance)
    
    # 3. Context filteren op relevantie
    # Stel een drempelwaarde in: bijv. een distance > 0.5 is vaak niet relevant meer
    relevant_docs = [row[0] for row in rows if row[1] < 0.5] 
    
    if not relevant_docs:
        context = "Geen relevante informatie gevonden in de lokale database."
        found_in_db = False
    else:
        context = "\n".join([doc.content for doc in relevant_docs])
        found_in_db = True
    
    # 4. Antwoord genereren
    answer = await generate_answer(request.request, context)

    # 5. Uitgebreide response teruggeven
    return {
        "answer": answer,
        "is_from_database": found_in_db,
        "sources": [
            {
                "id": doc.id, 
                "content": doc.content[:200] + "...", # Eerste stukje tekst
                "distance": round(float(dist), 4)
            } for doc, dist in rows
        ]
    }