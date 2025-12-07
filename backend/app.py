import os
import uuid
from urllib.parse import quote

from flask import Flask, request, jsonify
from flask_cors import CORS
import requests
import chromadb
from chromadb.config import Settings

# ---------- Config ----------

# Google Gemini API key (from Google AI Studio / makersuite).
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")
if not GEMINI_API_KEY:
    print("⚠️ WARNING: GEMINI_API_KEY not set. Set it in Render dashboard.")

GEMINI_EMBED_URL = (
    "https://generativelanguage.googleapis.com/v1beta/models/"
    "text-embedding-004:embedContent"
)
GEMINI_CHAT_URL = (
    "https://generativelanguage.googleapis.com/v1beta/models/"
    "gemini-1.5-flash:generateContent"
)

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
    "Dessert",
]

# ---------- Flask App ----------

app = Flask(__name__)
CORS(app)

# ---------- Chroma Vector DB ----------

chroma_client = chromadb.PersistentClient(
    path="./chroma_db", settings=Settings(allow_reset=True)
)
collection = chroma_client.get_or_create_collection(name="cook_knowledge")


def embed_texts(texts):
    """
    Use Gemini text-embedding-004 to convert text -> vectors.
    """
    if not texts:
        return []

    embeddings = []
    for text in texts:
        try:
            payload = {
                "model": "models/text-embedding-004",
                "content": {"parts": [{"text": text}]},
            }
            resp = requests.post(
                GEMINI_EMBED_URL,
                params={"key": GEMINI_API_KEY},
                json=payload,
                timeout=20,
            )
            resp.raise_for_status()
            data = resp.json()
            vec = data["embedding"]["values"]
            embeddings.append(vec)
        except Exception as e:
            print(f"Embedding error: {e}")
            # simple fallback: zero vector of length 768
            if embeddings:
                dim = len(embeddings[0])
            else:
                dim = 768
            embeddings.append([0.0] * dim)

    return embeddings


def build_prompt_from_messages(messages):
    """
    Turn list of {role, content} into a single text prompt for Gemini.
    """
    lines = []
    for m in messages:
        role = m.get("role", "user")
        content = m.get("content", "")
        if not content:
            continue
        if role == "system":
            prefix = "System"
        elif role == "assistant":
            prefix = "Assistant"
        else:
            prefix = "User"
        lines.append(f"{prefix}: {content}")
    return "\n\n".join(lines)


@app.get("/")
def health():
    return jsonify(
        {"status": "ok", "message": "CookWise RAG backend (Gemini) is running."}
    )


@app.post("/ingest")
def ingest():
    """
    Fetch cooking-related content from Wikipedia topics,
    embed them, and store in Chroma vector DB.

    If Wikipedia cannot be reached or returns no data,
    fall back to a built-in cooking knowledge dataset.
    """
    docs = []
    metas = []
    ids = []

    # Try online data source: Wikipedia
    for title in TOPICS:
        try:
            url = (
                "https://en.wikipedia.org/api/rest_v1/page/summary/"
                f"{quote(title)}"
            )
            resp = requests.get(url, timeout=10)

            if resp.status_code != 200:
                print(f"Wiki {title}: status {resp.status_code}")
                continue

            data = resp.json()
            text = data.get("extract", "")
            page_url = (
                data.get("content_urls", {})
                .get("desktop", {})
                .get("page", "")
            )

            if not text:
                print(f"Wiki {title}: empty extract")
                continue

            docs.append(text)
            metas.append({"title": title, "source": page_url})
            ids.append(str(uuid.uuid4()))
        except Exception as e:
            print(f"Error fetching {title}: {e}")
            continue

    # Fallback: built-in dataset if Wikipedia gave nothing
    if not docs:
        print("No docs from Wikipedia. Using fallback cooking dataset.")
        fallback_docs = [
            "Italian cuisine is known for pasta, pizza, olive oil, tomatoes, herbs like basil and oregano, and simple recipes that focus on fresh ingredients.",
            "Indian cuisine uses a large variety of spices such as turmeric, cumin, coriander, garam masala, chili, and often combines them in layered curries.",
            "Baking is a dry-heat cooking method that uses an oven. Common baked foods include bread, cakes, cookies, and pies.",
            "Grilling cooks food over direct high heat, usually from below, creating smoky flavors and grill marks on meat and vegetables.",
            "Food safety in the kitchen includes washing hands, avoiding cross-contamination of raw meat and vegetables, cooking meat to safe internal temperatures, and refrigerating leftovers quickly.",
            "Mediterranean cuisine emphasizes vegetables, legumes, whole grains, olive oil, fish, and moderate dairy, and is associated with heart-health benefits.",
            "Vegan cooking avoids all animal products, including meat, dairy, eggs, and honey, and often uses beans, lentils, tofu, and nuts for protein.",
            "Desserts include sweet dishes like cakes, ice cream, custards, and pastries, usually served at the end of a meal.",
        ]
        for i, text in enumerate(fallback_docs):
            docs.append(text)
            metas.append(
                {"title": f"Fallback doc {i+1}", "source": "local-fallback"}
            )
            ids.append(str(uuid.uuid4()))

    # Embed and store in Chroma
    embeddings = embed_texts(docs)
    collection.add(
        documents=docs, metadatas=metas, ids=ids, embeddings=embeddings
    )

    return jsonify({"status": "ok", "documents_added": len(docs)})


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

    if collection.count() == 0:
        return jsonify(
            {"error": "Knowledge base is empty. Call /ingest first."}
        ), 400

    # 1) Embed the query
    query_embedding = embed_texts([user_message])[0]

    # 2) Retrieve top-k similar docs
    results = collection.query(query_embeddings=[query_embedding], n_results=5)
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
                "If something is unknown, say you are not sure "
                "instead of making it up."
            ),
        },
        {
            "role": "system",
            "content": f"Knowledge base context:\n{context_str}",
        },
    ]

    for h in history[-6:]:
        role = h.get("role")
        content = h.get("content")
        if role in ("user", "assistant") and content:
            messages.append({"role": role, "content": content})

    messages.append({"role": "user", "content": user_message})

    prompt_text = build_prompt_from_messages(messages)

    # 4) Call Gemini for generation
    try:
        payload = {
            "contents": [
                {
                    "parts": [
                        {
                            "text": prompt_text,
                        }
                    ]
                }
            ]
        }
        resp = requests.post(
            GEMINI_CHAT_URL,
            params={"key": GEMINI_API_KEY},
            json=payload,
            timeout=30,
        )
        resp.raise_for_status()
        data = resp.json()
        answer = (
            data["candidates"][0]["content"]["parts"][0]["text"].strip()
        )
    except Exception as e:
        print(f"Gemini chat error: {e}")
        return jsonify({"error": "LLM generation failed"}), 500

    return jsonify({"answer": answer, "sources": metadatas})


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
