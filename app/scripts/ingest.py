import pandas as pd
from sqlmodel import Session
from database import engine
from models import Document
from core.llm import get_embedding

def ingest_csv(file_path: str, text_column: str):
    # 1. Lees CSV
    df = pd.read_csv(file_path)
    
    with Session(engine) as session:
        for _, row in df.iterrows():
            content = str(row[text_column])
            
            # 2. Genereer embedding voor de tekst in deze rij
            print(f"Verwerken: {content[:50]}...")
            vector = get_embedding(content)
            
            # 3. Maak SQLModel object
            # We slaan de hele rij op als metadata (JSON)
            doc = Document(
                content=content,
                embedding=vector,
                metadata=row.to_json()
            )
            
            session.add(doc)
        
        session.commit()
    print("Succesvol geïmporteerd!")

if __name__ == "__main__":
    # Gebruik: geef het pad naar je csv en de naam van de kolom met de tekst
    ingest_csv("../app/data/documents.csv", text_column="omschrijving")