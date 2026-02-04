from pgvector.sqlalchemy import Vector
from pydantic import ConfigDict, BaseModel
from sqlmodel import SQLModel, Field, Column
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
    sources: List[str]
    
class MyModel(BaseModel):
    test: str
    numbers: List[int]

class Hero(SQLModel, table=True):
    id: int | None = Field(default=None, primary_key=True)
    name: str = Field(index=True)
    age: int | None = Field(default=None, index=True)
    secret_name: str