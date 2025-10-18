import argparse
from textwrap import dedent
from typing import List
from openai import OpenAI

from config import OPENAI_API_KEY, OPENAI_CHAT_MODEL
from retrieve import retrieve_similar

client = OpenAI(api_key=OPENAI_API_KEY)

def build_prompt(question: str, contexts: List[str]) -> str:
    joined = "\n\n---\n\n".join(contexts)
    system = dedent(f"""\
    You are an assistant answering questions strictly based on the provided student handbook excerpts.
    - If the answer isn't present, say you don't know and suggest where it might be in the handbook.
    - Quote or paraphrase the exact relevant policy language where useful.
    - Keep answers concise and cite chunk numbers like [Chunk #].
    """)
    user = dedent(f"""\
    Question: {question}

    Context (top matches from the handbook):
    {joined}

    Reply with the best possible answer grounded in the context. If unsure, say so.
    """)
    # We'll put system in messages[0] and user in messages[1]
    return system, user

def format_contexts(retrieved):
    # annotate contexts with chunk numbers for lightweight "citations"
    contexts = []
    for _, content, ordinal, score in retrieved:
        header = f"[Chunk {ordinal} | score={score:.4f}]\n"
        contexts.append(header + content)
    return contexts

def answer(question: str) -> str:
    retrieved = retrieve_similar(question)
    contexts = format_contexts(retrieved)
    system, user = build_prompt(question, contexts)

    resp = client.chat.completions.create(
        model=OPENAI_CHAT_MODEL,
        temperature=0.2,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
    )
    return resp.choices[0].message.content

def main():
    parser = argparse.ArgumentParser(description="Ask a question about the handbook.")
    parser.add_argument("question", type=str, help="Your question")
    args = parser.parse_args()

    print(answer(args.question))

if __name__ == "__main__":
    main()
