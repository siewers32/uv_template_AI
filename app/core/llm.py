from openai import OpenAI
import os
from dotenv import load_dotenv

load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

def get_embedding(text: str) -> list[float]:
    response = client.embeddings.create(
        input=[text.replace("\n", " ")],
        model=os.getenv("TEXT_MODEL")
    )
    return response.data[0].embedding

def generate_answer(question: str, context: str) -> str:
    prompt = f"""Gebruik de onderstaande context om de vraag te beantwoorden. 
    Context: {context}
    Vraag: {question}
    Antwoord:"""
    
    response = client.chat.completions.create(
        model=os.getenv("LLM_MODEL"),
        messages=[{"role": "user", "content": prompt}]
    )
    return response.choices[0].message.content