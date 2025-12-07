// TODO: replace this after backend deployment:
const API_BASE = "https://YOUR-RENDER-BACKEND-URL.onrender.com";

let history = [];

const chatEl = document.getElementById("chat");
const statusEl = document.getElementById("status");
const ingestBtn = document.getElementById("ingestBtn");
const sendBtn = document.getElementById("sendBtn");
const messageInput = document.getElementById("message");

function addMessage(role, content) {
  history.push({ role, content });
  const div = document.createElement("div");
  div.className = `msg ${role}`;
  div.innerText = content;
  chatEl.appendChild(div);
  chatEl.scrollTop = chatEl.scrollHeight;
}

function setStatus(text) {
  statusEl.innerText = text;
}

async function ingestKnowledge() {
  ingestBtn.disabled = true;
  setStatus("Loading knowledge from Wikipedia...");
  try {
    const res = await fetch(`${API_BASE}/ingest`, { method: "POST" });
    const json = await res.json();
    if (json.status === "ok") {
      setStatus(`Loaded ${json.documents_added} documents.`);
    } else {
      setStatus(json.message || "Ingestion failed.");
    }
  } catch (err) {
    console.error(err);
    setStatus("Error during ingestion. Check console.");
  } finally {
    ingestBtn.disabled = false;
  }
}

async function sendMessage() {
  const text = messageInput.value.trim();
  if (!text) return;

  addMessage("user", text);
  messageInput.value = "";
  setStatus("Thinking...");
  sendBtn.disabled = true;

  try {
    const res = await fetch(`${API_BASE}/chat`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        message: text,
        history: history,
      }),
    });

    const json = await res.json();
    if (json.answer) {
      addMessage("assistant", json.answer);
      setStatus("");
    } else if (json.error) {
      addMessage("assistant", "Error: " + json.error);
      setStatus(json.error);
    } else {
      addMessage("assistant", "Unexpected response from server.");
      setStatus("Unexpected response.");
    }
  } catch (err) {
    console.error(err);
    addMessage("assistant", "Error calling backend.");
    setStatus("Error calling backend.");
  } finally {
    sendBtn.disabled = false;
  }
}

ingestBtn.addEventListener("click", ingestKnowledge);
sendBtn.addEventListener("click", sendMessage);
messageInput.addEventListener("keydown", (e) => {
  if (e.key === "Enter" && !e.shiftKey) {
    e.preventDefault();
    sendMessage();
  }
});
