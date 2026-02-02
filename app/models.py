from pgvector.sqlalchemy import Vector
from sqlmodel import SQLModel, Field, Column
from typing import Optional, List

class DocumentBase(SQLModel):
    content: str
    metadata: str = Field(default="{}")

class Document(DocumentBase, table=True):
    id: Optional[int] = Field(default=None, primary_key=True)
    # 1536 is de standaard voor OpenAI 'text-embedding-3-small'
    embedding: List[float] = Field(sa_column=Column(Vector(1536)))

class QueryRequest(SQLModel):
    question: str

class QueryResponse(SQLModel):
    answer: str
    sources: List[str]





