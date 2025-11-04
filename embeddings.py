import time
from typing import List
from openai import OpenAI
from config import OPENAI_API_KEY, OPENAI_EMBED_MODEL

client = OpenAI(api_key=OPENAI_API_KEY)

def embed_texts(texts: List[str]) -> List[List[float]]:
    for attempt in range(5):
        try:
            resp = client.embeddings.create(model=OPENAI_EMBED_MODEL, input=texts)
            return [d.embedding for d in resp.data]
        except Exception as e:
            if attempt == 4:
                raise
            sleep_time = 1.5 * (2 ** attempt)
            time.sleep(sleep_time)

    raise RuntimeError("embed_texts gone horribly wrong??")
