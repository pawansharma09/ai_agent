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

# Load API keys from Streamlit secrets
DG_KEY = st.secrets.get("DEEPGRAM_API_KEY")
SERPER_KEY = st.secrets.get("SERPER_API_KEY")
GEN_KEY = st.secrets.get("GEMINI_API_KEY")
# PC_KEY is not used, so we can comment it out or handle it similarly if needed
# PC_KEY = st.secrets.get("PINECONE_API_KEY") 

if not all([DG_KEY, SERPER_KEY, GEN_KEY]):
    st.warning("Please set DEEPGRAM_API_KEY, SERPER_API_KEY, and GEMINI_API_KEY in your Streamlit secrets.")
    st.stop()

# ---------- INIT ----------
# Use caching for models to avoid reloading on each run
@st.cache_resource
def load_models():
    embed_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    genai.configure(api_key=GEN_KEY)
    dg = DeepgramClient(DG_KEY)
    # Changed: Corrected the Gemini model name to a valid public model
    text_model = genai.GenerativeModel("gemini-1.5-flash-latest")
    return embed_model, dg, text_model

embed_model, dg, text_model = load_models()

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

def chunk_pages(pages, chunk_size=500):
    chunks = []
    for p in pages:
        words = p["text"].split()
        for i in range(0, len(words), chunk_size):
            seg = " ".join(words[i:i + chunk_size])
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
    if not st.session_state.vectors:
        return []
    q_emb = embed_model.encode(q)
    scores = []
    for emb, meta in st.session_state.vectors:
        sim = np.dot(q_emb, emb) / (np.linalg.norm(q_emb) * np.linalg.norm(emb))
        scores.append((sim, meta))
    scores.sort(reverse=True, key=lambda x: x[0])
    return [{"score": s, "text": m["text"], "page": m["page"]} for s, m in scores[:top_k]]

def web_search(q):
    try:
        r = requests.post(
            "https://google.serper.dev/search",
            headers={"X-API-KEY": SERPER_KEY, "Content-Type": "application/json"},
            json={"q": q},
            timeout=15
        )
        r.raise_for_status()  # Will raise an exception for 4XX/5XX errors
        j = r.json()
        lines = []
        for i in j.get("organic", [])[:3]:
            lines.append(f"{i.get('title','')} — {i.get('link','')}")
        return "\n".join(lines)
    except requests.exceptions.RequestException as e:
        st.error(f"Web search failed: {e}")
        return "No web results."

def call_gemini(prompt):
    res = text_model.generate_content(prompt)
    return res.text

def pdf_page_image(pdf_bytes, page_num):
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    page = doc.load_page(page_num - 1)  # page_num is 1-based, load_page is 0-based
    pix = page.get_pixmap(matrix=fitz.Matrix(2, 2))  # Increase resolution
    return Image.open(io.BytesIO(pix.tobytes("png")))

def transcribe(audio_bytes, mimetype):
    # Changed: Mimetype is now passed as an argument
    source = {"buffer": audio_bytes, "mimetype": mimetype}
    options = {"smart_format": True, "model": "nova-2", "language": "en-US"}
    res = dg.listen.prerecorded.v("1").transcribe_file(source, options)
    return res["results"]["channels"][0]["alternatives"][0]["transcript"]

# ---------- UI ----------
st.sidebar.header("1. Upload Document")
uploaded = st.sidebar.file_uploader("📄 Upload PDF (with text & images)", type=["pdf"])
if uploaded:
    with st.spinner("Processing PDF..."):
        pdf_data = uploaded.read()
        pages = extract_pdf(pdf_data)
        chunks = chunk_pages(pages)
        index_local(chunks)
        st.session_state["pdf_data"] = pdf_data
        st.session_state["pages"] = pages
        st.sidebar.success(f"Indexed {len(chunks)} chunks from {uploaded.name}.")

st.header("2. Ask with your voice")
audio = st.file_uploader("Upload or record a question (wav/mp3)", type=["wav", "mp3"], key="audio_uploader")

if audio and st.session_state.get("vectors"):
    audio_bytes = audio.read()
    st.audio(audio_bytes)

    with st.spinner("Transcribing your question..."):
        # Changed: Pass the dynamic mimetype from the uploaded file
        text = transcribe(audio_bytes, audio.type)
    st.write(f"**You said:** {text}")

    with st.spinner("Retrieving context and generating answer..."):
        pdf_hits = search_local(text)
        web_hits = web_search(text)
        
        prompt = f"""
        Answer the following question based only on the provided context from the PDF excerpts and web results.
        Cite the PDF page number for any information you use from it, like this: (Page X).

        Question: {text}

        PDF excerpts:
        {json.dumps(pdf_hits, indent=2)}

        Web results:
        {web_hits}
        """

        ans = call_gemini(prompt)
        st.subheader("💬 Answer")
        st.write(ans)

        if pdf_hits:
            st.subheader("📖 PDF Citations")
            for h in pdf_hits:
                st.markdown(f"**Page {h['page']}** — similarity {h['score']:.2f}")
                img = pdf_page_image(st.session_state["pdf_data"], h["page"])
                st.image(img, use_column_width=True, caption=f"Page {h['page']}")
else:
    st.info("Upload a PDF in the sidebar and then ask your question by voice!")

st.sidebar.caption("Powered by: Deepgram • Serper • Gemini • Sentence Transformers")
