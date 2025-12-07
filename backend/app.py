import os
import math
import uuid
from flask import Flask, request, jsonify
from flask_cors import CORS

# ---------- CONFIG ----------

GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")
if not GEMINI_API_KEY:
    print("⚠️ WARNING: GEMINI_API_KEY not set. Set it in Render dashboard.")

GEMINI_CHAT_URL = (
    "https://generativelanguage.googleapis.com/v1beta/models/"
    "gemini-1.5-flash:generateContent"
)

app = Flask(__name__)
CORS(app)

# In-memory "vector store"
DOCUMENTS = []       # list of strings
METADATAS = []       # list of dicts
EMBEDDINGS = []      # list of vectors (list[float])


# ---------- SIMPLE LOCAL EMBEDDINGS & RETRIEVAL ----------

def embed_text(text: str):
    """
    Very lightweight local embedding:
    We turn a text into a small numeric feature vector.
    No network, no external API, fast for free-tier.
    """
    t = text.lower()
    length = len(t)
    words = t.split()
    word_count = len(words)
    vowels = sum(t.count(v) for v in "aeiou")
    consonants = sum(c.isalpha() for c in t) - vowels
    avg_word_len = length / max(word_count, 1)

    return [
        float(length),
        float(word_count),
        float(vowels),
        float(consonants),
        float(avg_word_len),
    ]


def cosine_sim(a, b):
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(x * x for x in b))
    if na == 0 or nb == 0:
        return 0.0
    return dot / (na * nb)


def top_k_docs(query, k=5):
    if not DOCUMENTS:
        return [], []

    q_emb = embed_text(query)
    scored = []
    for i, emb in enumerate(EMBEDDINGS):
        score = cosine_sim(q_emb, emb)
        scored.append((score, i))

    scored.sort(reverse=True, key=lambda x: x[0])
    top = scored[:k]

    docs = [DOCUMENTS[i] for _, i in top]
    metas = [METADATAS[i] for _, i in top]
    return docs, metas


# ---------- ROUTES ----------

@app.get("/")
def health():
    return jsonify({
        "status": "ok",
        "message": "CookWise RAG backend (lightweight) is running."
    })


@app.post("/ingest")
def ingest():
    """
    Load a built-in cooking knowledge dataset, compute local embeddings,
    store them in in-memory lists. Fast and stable on free-tier.
    """
    global DOCUMENTS, METADATAS, EMBEDDINGS

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

    DOCUMENTS = []
    METADATAS = []
    EMBEDDINGS = []

    for i, text in enumerate(fallback_docs):
        DOCUMENTS.append(text)
        METADATAS.append({
            "id": str(uuid.uuid4()),
            "title": f"Cooking doc {i+1}",
            "source": "local-fallback"
        })
        EMBEDDINGS.append(embed_text(text))

    return jsonify({
        "status": "ok",
        "documents_added": len(DOCUMENTS)
    })


@app.post("/chat")
def chat():
    """
    RAG chat endpoint.
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

    if not DOCUMENTS:
        return jsonify({"error": "Knowledge base is empty. Call /ingest first."}), 400

    # 1) Retrieve top docs using local embeddings
    docs, metas = top_k_docs(user_message, k=5)

    context_blocks = []
    for text, meta in zip(docs, metas):
        title = meta.get("title", "Unknown")
        source = meta.get("source", "")
        block = f"Title: {title}\nSource: {source}\nContent: {text}"
        context_blocks.append(block)

    context_str = "\n\n---\n\n".join(context_blocks)

    # 2) Build prompt for Gemini
    lines = [
        "System: You are CookWise, an expert cooking assistant.",
        "System: Answer ONLY using the provided context. "
        "If the answer is not in the context, say you are not sure.",
        f"System: Context:\n{context_str}",
    ]

    for h in history[-6:]:
        role = h.get("role")
        content = h.get("content")
        if not content:
            continue
        if role == "assistant":
            lines.append(f"Assistant: {content}")
        else:
            lines.append(f"User: {content}")

    lines.append(f"User: {user_message}")
    prompt_text = "\n\n".join(lines)

    try:
        payload = {
            "contents": [
                {
                    "parts": [
                        {"text": prompt_text}
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
        answer = data["candidates"][0]["content"]["parts"][0]["text"].strip()
    except Exception as e:
        print(f"Gemini chat error: {e}")
        return jsonify({"error": "LLM generation failed"}), 500

    return jsonify({"answer": answer, "sources": metas})


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
