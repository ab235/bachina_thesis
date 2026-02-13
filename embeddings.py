import time
from typing import List, Optional
from openai import OpenAI
from config import OPENAI_API_KEY, OPENAI_EMBED_MODEL

client = OpenAI(api_key=OPENAI_API_KEY)

def embed_texts(texts: List[str], model_name: Optional[str] = None) -> List[List[float]]:
    for attempt in range(5):
        try:
            model = model_name or OPENAI_EMBED_MODEL
            resp = client.embeddings.create(model=model, input=texts)
            return [d.embedding for d in resp.data]
        except Exception as e:
            if attempt == 4:
                raise
            sleep_time = 1.5 * (2 ** attempt)
            time.sleep(sleep_time)

    raise RuntimeError("embed_texts gone horribly wrong??")
