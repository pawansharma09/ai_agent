# app.py

import streamlit as st
import os
import fitz  # PyMuPDF
import base64
import pinecone
from PIL import Image
import io
from dotenv import load_dotenv

# --- LangChain Imports ---
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.embeddings import HuggingFaceEmbeddings # <-- NEW: Using Hugging Face
from langchain.agents import create_tool_calling_agent, AgentExecutor
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import AIMessage, HumanMessage
from langchain.tools.retriever import create_retriever_tool
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_pinecone import PineconeVectorStore
from langchain_community.utilities import GoogleSerperAPIWrapper

# --- PAGE CONFIGURATION & API KEY LOADING ---
st.set_page_config(page_title="Agentic RAG Assistant", layout="wide")
load_dotenv()

# Check for API keys in environment variables
required_keys = ["GOOGLE_API_KEY", "PINECONE_API_KEY", "SERPER_API_KEY", "DEEPGRAM_API_KEY"]
missing_keys = [key for key in required_keys if not os.getenv(key)]

if missing_keys:
    st.error(f"Missing API keys for: {', '.join(missing_keys)}. Please add them to your .env file.")
    st.stop()

# Initialize Pinecone
pinecone_api_key = os.getenv("PINECONE_API_KEY")
pinecone.init(api_key=pinecone_api_key)
INDEX_NAME = "multimodal-rag-index-hf" # Using a new index name for the new embedding model

# --- CRITICAL CHANGE: Update Pinecone index dimension for Hugging Face model ---
# all-MiniLM-L6-v2 has a dimension of 384
DIMENSION = 384
if INDEX_NAME not in pinecone.list_indexes():
    st.info(f"Creating new Pinecone index '{INDEX_NAME}' with dimension {DIMENSION}...")
    pinecone.create_index(name=INDEX_NAME, dimension=DIMENSION, metric="cosine")
    st.success(f"Pinecone index '{INDEX_NAME}' created successfully.")

# --- CORE FUNCTIONS ---

@st.cache_resource(show_spinner="Analyzing Document...")
def process_document(uploaded_file):
    """
    Processes the uploaded PDF: extracts text and images, generates image summaries,
    creates embeddings using Hugging Face, and upserts them into Pinecone.
    """
    file_bytes = uploaded_file.getvalue()
    doc = fitz.open(stream=file_bytes, filetype="pdf")
    
    texts = []
    images = []
    
    # Use Gemini 1.5 Pro for vision tasks
    vision_model = ChatGoogleGenerativeAI(model="gemini-2.5-flash", max_tokens=1024)
    # Use local Hugging Face model for embeddings
    embeddings_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

    for page_num, page in enumerate(doc):
        page_text = page.get_text()
        if page_text:
            texts.append({"text": page_text, "metadata": {"page": page_num + 1}})

        for img_index, img in enumerate(page.get_images(full=True)):
            xref = img[0]
            base_image = doc.extract_image(xref)
            image_bytes = base_image["image"]
            
            try:
                img_summary_payload = [
                    HumanMessage(content=[
                        {"type": "text", "text": "Describe this image in detail. What information does it convey?"},
                        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64.b64encode(image_bytes).decode()}"}}
                    ])
                ]
                summary_response = vision_model.invoke(img_summary_payload)
                images.append({"text": summary_response.content, "metadata": {"page": page_num + 1, "image_index": img_index}})
            except Exception as e:
                st.warning(f"Could not analyze an image on page {page_num+1}. Skipping. Error: {e}")

    combined_docs = texts + images
    
    # Upsert to Pinecone
    vectorstore = PineconeVectorStore.from_texts(
        texts=[d['text'] for d in combined_docs],
        embedding=embeddings_model,
        metadatas=[d['metadata'] for d in combined_docs],
        index_name=INDEX_NAME
    )
    return vectorstore, file_bytes

def get_page_image(file_bytes, page_number):
    """Renders a specific page of the PDF as a PNG image."""
    doc = fitz.open(stream=file_bytes, filetype="pdf")
    page = doc.load_page(page_number - 1)
    pix = page.get_pixmap()
    img_bytes = pix.tobytes("png")
    return Image.open(io.BytesIO(img_bytes))

# --- AGENT SETUP ---
# Use Gemini 1.5 Pro as the agent's brain
llm = ChatGoogleGenerativeAI(model="gemini-1.5-pro-latest", temperature=0)

prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant. You answer questions based on the provided document and can also search the web. For any information retrieved from the document, you MUST cite the page number. When a user asks a question, first decide if you should search the web or the document. If the user asks for current events or general knowledge, use the web search. If the user asks about the content of the uploaded document, use the retriever."),
    ("human", "{input}"),
    ("placeholder", "{agent_scratchpad}"),
])

# Tool 1: Web Search
search = GoogleSerperAPIWrapper()
web_search_tool = st.cache_resource(lambda: TavilySearchResults(api_wrapper=search))()
web_search_tool.name = "web_search"
web_search_tool.description = "Searches the web for real-time information."

# --- UI RENDERING ---
st.title("🎙️ Voice-Enabled Agentic RAG Assistant")
st.markdown("Upload a PDF and ask questions using your voice or by typing. The assistant will search the document (text and images) and the web, providing cited answers and source pages.")

if "messages" not in st.session_state:
    st.session_state.messages = []

with st.sidebar:
    st.header("1. Upload Document")
    uploaded_file = st.file_uploader("Choose a PDF file", type="pdf")
    st.header("2. Ask a Question")
    st.markdown("Use the text input or the (simulated) voice button in the main chat interface.")

if not uploaded_file:
    st.info("Please upload a PDF document to begin.")
    st.stop()

vectorstore, file_bytes = process_document(uploaded_file)
retriever = vectorstore.as_retriever(search_kwargs={'k': 3})
retriever_tool = create_retriever_tool(retriever, "document_search", "Searches the uploaded PDF document for relevant information.")

tools = [web_search_tool, retriever_tool]

agent = create_tool_calling_agent(llm, tools, prompt)
agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if "citations" in message:
            for citation in message["citations"]:
                st.image(citation, caption=f"Source: Page {message['page_number']}", width=400)

if st.button("🎤 (Simulated Voice Input)"):
    st.session_state.user_prompt = st.text_input("I'm listening...", key=f"voice_input_{len(st.session_state.messages)}")

prompt_input = st.chat_input("Or type your question here...")
user_prompt = prompt_input or st.session_state.get('user_prompt')

if user_prompt:
    st.session_state.messages.append({"role": "user", "content": user_prompt})
    with st.chat_message("user"):
        st.markdown(user_prompt)

    with st.chat_message("assistant"):
        with st.spinner("Thinking... (The agent is deciding which tool to use)"):
            response = agent_executor.invoke({"input": user_prompt})
            st.markdown(response["output"])
            
            citations = []
            page_numbers = set()
            if response.get("intermediate_steps"):
                for step in response["intermediate_steps"]:
                    if step[0].tool == 'document_search':
                        for doc in step[1]:
                            page_numbers.add(doc.metadata['page'])
            
            if page_numbers:
                st.markdown("---")
                st.subheader("Citations from Document:")
                for page_num in sorted(list(page_numbers)):
                    img = get_page_image(file_bytes, page_num)
                    citations.append(img)
                    st.image(img, caption=f"Source: Page {page_num}", width=500)
            
            assistant_message = {"role": "assistant", "content": response["output"]}
            if citations:
                assistant_message["citations"] = citations
                assistant_message["page_number"] = sorted(list(page_numbers))[0]
            
            st.session_state.messages.append(assistant_message)
    
    if 'user_prompt' in st.session_state:
        del st.session_state['user_prompt']
