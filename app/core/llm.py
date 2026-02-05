from openai import AsyncOpenAI
import os
from dotenv import load_dotenv

load_dotenv()

client = AsyncOpenAI(
    api_key=os.getenv("OPENAI_API_KEY"),
    base_url=os.getenv("API_URL")
)

print(os.getenv("TEXT_MODEL"))

async def get_embedding(text: str) -> list[float]:
    response = await client.embeddings.create(
        input=[text.replace("\n", " ")],
        model=os.getenv("TEXT_MODEL")
    )
    return response.data[0].embedding

async def generate_answer(question: str, context: str) -> str:
    prompt = f""""Gebruik uitsluitend de verstrekte context om de vraag te beantwoorden. Als het antwoord niet in de context staat, zeg dan expliciet dat je het niet in de database hebt gevonden, maar antwoord daarna op basis van je eigen kennis."
    Gebruik de onderstaande context om de vraag te beantwoorden. 
    Context: {context}
    Vraag: {question}
    Antwoord:"""
    
    response = await client.chat.completions.create(
        model=os.getenv("LLM_MODEL"),
        messages=[{"role": "user", "content": prompt}]
    )
    return response.choices[0].message.content