from pgvector.sqlalchemy import Vector
from sqlmodel import SQLModel, Field, Column
from pydantic import BaseModel
from typing import Optional, List

class Document(SQLModel, table=True):
    id: Optional[int] = Field(default=None, primary_key=True)
    content: str = Field(index=True)
    # We hernoemen 'metadata' naar 'extra_info' om conflicten te voorkomen
    extra_info: str = Field(default="{}") 
    embedding: List[float] = Field(sa_column=Column(Vector(1024)))

class QueryRequest(SQLModel):
    request: str

class QueryResponse(SQLModel):
    answer: str
    is_from_database: bool
    sources: List[SourceDetail]

class SourceDetail(BaseModel):
    id: Optional[int]
    content: str
    distance: float  # De afstandsscore uit de vector search


