from openai import OpenAI
import os
from dotenv import load_dotenv

load_dotenv()

client = OpenAI(
    api_key=os.getenv("OPENAI_API_KEY"),
    base_url="http://localhost:1234/v1"
)

print(os.getenv("TEXT_MODEL"))

def get_embedding(text: str) -> list[float]:
    response = client.embeddings.create(
        input=[text.replace("\n", " ")],
        model=os.getenv("TEXT_MODEL")
    )
    return response.data[0].embedding

get_embedding("glq;werwe;wer;wer") 