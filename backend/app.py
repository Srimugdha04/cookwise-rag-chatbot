import os
import uuid
from urllib.parse import quote

from flask import Flask, request, jsonify
from flask_cors import CORS
import requests
import chromadb
from chromadb.config import Settings
from openai import OpenAI

# ---------- Config ----------
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    print("⚠️ WARNING: OPENAI_API_KEY not set. Set it in Render dashboard.")

client = OpenAI(api_key=OPENAI_API_KEY)

# Wikipedia topics for cooking / cuisine – our online data source
TOPICS = [
    "Italian cuisine",
    "Indian cuisine",
    "Chinese cuisine",
    "Baking",
    "Grilling",
    "Spices",
    "Veganism",
    "Food safety",
    "Mediterranean cuisine",
    "Dessert"
]

# ---------- Flask App ----------
app = Flask(__name__)
CORS(app)

# ---------- Chroma Vector DB ----------
# Persistent client so data survives restarts (on Render disk)
chroma_client = chromadb.PersistentClient(
    path="./chroma_db",
    settings=Settings(allow_reset=True)
)

collection = chroma_client.get_or_create_collection(name="cook_knowledge")


def embed_texts(texts):
    """
    Use OpenAI embeddings to convert text -> vector.
    """
    if not texts:
        return []

    response = client.embeddings.create(
        model="text-embedding-3-small",
        input=texts
    )
    return [d.embedding for d in response.data]


# ---------- Routes ----------

@app.get("/")
def health():
    return jsonify({
        "status": "ok",
        "message": "CookWise RAG backend is running."
    })


@app.post("/ingest")
def ingest():
    """
    Fetch cooking-related content from Wikipedia topics,
    embed them, and store in Chroma vector DB.
    """
    docs = []
    metas = []
    ids = []

    for title in TOPICS:
        try:
            url = f"https://en.wikipedia.org/api/rest_v1/page/summary/{quote(title)}"
            resp = requests.get(url, timeout=10)
            if resp.status_code != 200:
                continue

            data = resp.json()
            text = data.get("extract", "")
            page_url = (
                data.get("content_urls", {})
                .get("desktop", {})
                .get("page", "")
            )

            if not text:
                continue

            docs.append(text)
            metas.append({"title": title, "source": page_url})
            ids.append(str(uuid.uuid4()))

        except Exception as e:
            print(f"Error fetching {title}: {e}")
            continue

    if not docs:
        return jsonify({
            "status": "error",
            "message": "No documents fetched from Wikipedia."
        }), 500

    embeddings = embed_texts(docs)
    collection.add(
        documents=docs,
        metadatas=metas,
        ids=ids,
        embeddings=embeddings
    )

    return jsonify({
        "status": "ok",
        "documents_added": len(docs)
    })


@app.post("/chat")
def chat():
    """
    RAG Chat endpoint.
    Body:
    {
      "message": "How to make Italian pasta?",
      "history": [ { "role": "user"/"assistant", "content": "..." }, ... ]
    }
    """
    data = request.get_json(force=True)
    user_message = data.get("message", "").strip()
    history = data.get("history", [])

    if not user_message:
        return jsonify({"error": "message is required"}), 400

    # Ensure we have some documents ingested
    count = collection.count()
    if count == 0:
        return jsonify({
            "error": "Knowledge base is empty. Call /ingest first."
        }), 400

    # 1) Embed the query
    query_embedding = embed_texts([user_message])[0]

    # 2) Retrieve top-k similar docs
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=5
    )

    docs = results["documents"][0]
    metadatas = results["metadatas"][0]

    # Build context string
    context_blocks = []
    for text, meta in zip(docs, metadatas):
        title = meta.get("title", "Unknown")
        source = meta.get("source", "")
        block = f"Title: {title}\nSource: {source}\nContent: {text}"
        context_blocks.append(block)

    context_str = "\n\n---\n\n".join(context_blocks)

    # 3) Build chat messages with multi-turn history
    messages = [
        {
            "role": "system",
            "content": (
                "You are CookWise, an expert cooking assistant. "
                "Answer ONLY using the provided context. "
                "If something is unknown, say you are not sure"
                " instead of making it up."
            )
        },
        {
            "role": "system",
            "content": f"Knowledge base context:\n{context_str}"
        }
    ]

    # Include last few turns of history
    for h in history[-6:]:
        role = h.get("role")
        content = h.get("content")
        if role in ("user", "assistant") and content:
            messages.append({"role": role, "content": content})

    messages.append({"role": "user", "content": user_message})

    # 4) Call LLM
    completion = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=messages,
        temperature=0.4
    )

    answer = completion.choices[0].message.content

    return jsonify({
        "answer": answer,
        "sources": metadatas
    })


if __name__ == "__main__":
    # For local testing only. On Render we use gunicorn.
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
