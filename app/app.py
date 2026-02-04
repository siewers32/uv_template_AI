from fastapi import Depends, FastAPI, Request, Form
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from sqlalchemy.ext.asyncio import AsyncSession
from sqlmodel import select
from database import engine, get_session, init_db
from models import Document, MyModel, QueryRequest, QueryResponse, Hero
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



    
@app.get("/")
# async def toon_formulier(request: Request):
#     return templates.TemplateResponse("vraag.html", {"request": request})
async def toon_formulier(request: Request):
    return request


# 2. De POST route: Om de ingevulde vraag te verwerken
def question_to_question_request(
        request: str = "Slaapkamer",
        content: str = "Dit is de content" ,
        extra_info: str = "Dit is extra info",
        embedding: list[float] = [1.0, 2.0, 3.0]
        ) -> QueryRequest:
    return QueryRequest(request=request, content=content, extra_info=extra_info, embedding=embedding)
    
def to_mymodel() -> MyModel:
    return MyModel(test="test", numbers=[1,2,3])


@app.post("/")
async def verwerk_formulier(
     session: AsyncSession = Depends(get_session),
     request: QueryRequest = Depends(question_to_question_request), 
     m: MyModel = Depends(to_mymodel)
):
    # 1. Vraag omzetten naar vector
    # Vergeet 'await' niet als get_embedding nu async is!
    query_vec = await get_embedding(request.request) 
    
    # # 2. Vector search in Postgres (Asynchroon)
    async with AsyncSession(engine) as session:
        statement = (
            select(Document)
            # Gebruik cosine_distance, l2_distance of max_inner_product
            .order_by(Document.embedding.cosine_distance(query_vec))
            .limit(3)
        )
        results = await session.execute(statement)
        documents = results.scalars().all()
    
    # # 3. Context samenstellen
    context = "\n".join([doc.content for doc in documents])
    
    # # 4. Antwoord genereren
    answer = await generate_answer(request.request, context)


    return {
        "answer": answer,
        "sources": [doc.content for doc in documents]
    }

