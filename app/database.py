from sqlmodel import create_engine, Session, SQLModel
from core.config import settings

engine = create_engine(settings.DATABASE_URL)

def init_db():
    with Session(engine) as session:
        session.exec_driver_sql("CREATE EXTENSION IF NOT EXISTS vector")
    SQLModel.metadata.create_all(engine)

