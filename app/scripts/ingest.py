import pandas as pd
import sys
import os
from sqlmodel import Session

# Pad toevoegen voor imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from database import engine, init_db
from models import Document
from core.llm import get_embedding

def run_ingestion(csv_path: str, text_col: str):
    init_db()
    df = pd.read_csv(csv_path, sep=';', encoding='utf-8')

    # DEBUG: Print de kolomnamen die Pandas wél ziet
    print(f"Beschikbare kolommen in CSV: {df.columns.tolist()}")  

    with Session(engine) as session:
        for _, row in df.iterrows():
            content = str(row[text_col])
            print(f"Verwerken: {content[:30]}...")
            
            # Hier gebruiken we nu 'extra_info' in plaats van 'metadata'
            doc = Document(
                content=content,
                embedding=get_embedding(content),
                extra_info=row.to_json() 
            )
            session.add(doc)
        session.commit()
    print("Klaar!")

if __name__ == "__main__":
    run_ingestion("data/data.csv", text_col="description")