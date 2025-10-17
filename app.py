import os, io, json, base64, requests, fitz
import streamlit as st
from PIL import Image
from sentence_transformers import SentenceTransformer
from deepgram import DeepgramClient
import google.generativeai as genai
import numpy as np

# ---------- CONFIG ----------
st.set_page_config(page_title="Agentic Voice RAG Bot", layout="wide")
st.title("🎙️ Multi-Modal Agentic RAG Chatbot")

# Load API keys (Pinecone optional)
DG_KEY = os.getenv("DEEPGRAM_API_KEY")
SERPER_KEY = os.getenv("SERPER_API_KEY")
GEN_KEY = os.getenv("GEMINI_API_KEY")
PC_KEY = os.getenv("PINECONE_API_KEY")  # not used for index, just kept for compatibility

if not all([DG_KEY, SERPER_KEY, GEN_KEY]):
    st.warning("Please set DEEPGRAM_API_KEY, SERPER_API_KEY, GEMINI_API_KEY before running.")
    st.stop()

# ---------- INIT ----------
embed_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
genai.configure(api_key=GEN_KEY)
dg = Deepgram(DG_KEY)

# Local in-memory store
if "vectors" not in st.session_state:
    st.session_state.vectors = []  # list of (embedding, metadata)

# ---------- HELPERS ----------
def extract_pdf(pdf_bytes):
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    pages = []
    for i, page in enumerate(doc):
        text = page.get_text("text")
        imgs = []
        for img in page.get_images(full=True):
            base = doc.extract_image(img[0])
            imgs.append(Image.open(io.BytesIO(base["image"])))
        pages.append({"num": i + 1, "text": text, "images": imgs})
    return pages

def chunk_pages(pages, chunk=500):
    chunks = []
    for p in pages:
        words = p["text"].split()
        for i in range(0, len(words), chunk):
            seg = " ".join(words[i:i + chunk])
            if seg.strip():
                chunks.append({"text": seg, "page": p["num"]})
    return chunks

def index_local(chunks):
    vecs = []
    for c in chunks:
        emb = embed_model.encode(c["text"])
        vecs.append((emb, c))
    st.session_state.vectors = vecs

def search_local(q, top_k=3):
    q_emb = embed_model.encode(q)
    scores = []
    for emb, meta in st.session_state.vectors:
        sim = np.dot(q_emb, emb) / (np.linalg.norm(q_emb) * np.linalg.norm(emb))
        scores.append((sim, meta))
    scores.sort(reverse=True, key=lambda x: x[0])
    return [{"score": s, "text": m["text"], "page": m["page"]} for s, m in scores[:top_k]]

def web_search(q):
    r = requests.post(
        "https://google.serper.dev/search",
        headers={"X-API-KEY": SERPER_KEY, "Content-Type": "application/json"},
        json={"q": q},
        timeout=15
    )
    if r.status_code == 200:
        j = r.json()
        lines = []
        for i in j.get("organic", [])[:3]:
            lines.append(f"{i.get('title','')} — {i.get('link','')}")
        return "\n".join(lines)
    return "No web results."

def call_gemini(prompt):
    model = genai.GenerativeModel("gemini-2.0-flash-exp")
    res = model.generate_content(prompt)
    return res.text

def pdf_page_image(pdf_bytes, page_num):
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    page = doc.load_page(page_num - 1)
    pix = page.get_pixmap(matrix=fitz.Matrix(2, 2))
    return Image.open(io.BytesIO(pix.tobytes("png")))

def transcribe(audio_bytes):
    res = dg.transcription.sync_prerecorded(
        {"buffer": audio_bytes, "mimetype": "audio/wav"},
        {"smart_format": True}
    )
    return res["results"]["channels"][0]["alternatives"][0]["transcript"]

# ---------- UI ----------
uploaded = st.sidebar.file_uploader("📄 Upload PDF (with text & images)", type=["pdf"])
if uploaded:
    pdf_data = uploaded.read()
    pages = extract_pdf(pdf_data)
    chunks = chunk_pages(pages)
    index_local(chunks)
    st.session_state["pdf_data"] = pdf_data
    st.session_state["pages"] = pages
    st.success(f"Indexed {len(chunks)} chunks from {len(pages)} pages.")

st.header("🎧 Ask with your voice")
audio = st.file_uploader("Upload or record question (wav/mp3)", type=["wav", "mp3"])

if audio and "vectors" in st.session_state and st.session_state.vectors:
    st.info("Transcribing your question...")
    text = transcribe(audio.read())
    st.write(f"**You said:** {text}")

    st.info("Retrieving context...")
    pdf_hits = search_local(text)
    web_hits = web_search(text)

    prompt = f"""
Answer this question using the provided data. Cite PDF pages as (Page X).

Question: {text}

PDF excerpts:
{json.dumps(pdf_hits, indent=2)}

Web results:
{web_hits}
"""
    st.info("Generating answer with Gemini 2.5 Flash...")
    ans = call_gemini(prompt)
    st.subheader("💬 Answer")
    st.write(ans)

    st.subheader("📖 Citations")
    for h in pdf_hits:
        st.markdown(f"**Page {h['page']}** — similarity {h['score']:.2f}")
        img = pdf_page_image(st.session_state["pdf_data"], h["page"])
        st.image(img, use_column_width=True)
else:
    st.caption("Upload a PDF and then ask your question by voice!")

st.sidebar.caption("Free-tier only: Deepgram • Serper • Gemini 2.5 Flash • Sentence Transformer (local)")
