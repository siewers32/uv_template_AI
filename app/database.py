from typing import AsyncGenerator
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
from sqlalchemy.orm import sessionmaker
from sqlmodel import SQLModel, text
from dotenv import load_dotenv
import os

# Laad de .env file
load_dotenv()
# engine = create_async_engine(os.getenv("DATABASE_URL"))
engine = create_async_engine("postgresql+asyncpg://rag:rag@db/rag")

# De sessie-maker configuratie
async_session_maker = sessionmaker(
    engine, class_=AsyncSession, expire_on_commit=False
)

async def init_db():
    async with engine.begin() as conn:
        # Extensie installeren
        await conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))
        # Tabellen aanmaken (dit moet in een asynchrone context)
        await conn.run_sync(SQLModel.metadata.create_all)

# Helper om een sessie te krijgen
async def get_session() -> AsyncGenerator[AsyncSession, None]:
    async with async_session_maker() as session:
        yield session