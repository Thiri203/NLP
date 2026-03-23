import os
import json
from pathlib import Path
import time
import torch
import numpy as np
import streamlit as st
from dotenv import load_dotenv
from groq import Groq
from transformers import AutoTokenizer, AutoModel

load_dotenv()

st.set_page_config(page_title="Chapter 8 Transformer QA", page_icon="🤖")
st.title("Chapter 8 Transformer QA Chatbot")
st.caption("Contextual Retrieval-based QA system for Chapter 8")

# ----------------------------
# Load Groq client
# ----------------------------
groq_api_key = os.getenv("GROQ_API_KEY")
if not groq_api_key:
    st.error("GROQ_API_KEY not found in environment.")
    st.stop()

client = Groq(api_key=groq_api_key)

# ----------------------------
# Load embedding model
# ----------------------------
@st.cache_resource
def load_embedding_model():
    model_name = "BAAI/bge-small-en-v1.5"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)
    model.eval()
    return tokenizer, model, device

embed_tokenizer, embed_model, device = load_embedding_model()

def get_embedding(text):
    inputs = embed_tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        padding=True,
        max_length=512
    )
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = embed_model(**inputs)

    embedding = outputs.last_hidden_state[:, 0, :].squeeze().cpu().numpy()
    return embedding

def cosine_similarity(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

# ----------------------------
# Load contextual chunks
# ----------------------------
@st.cache_resource
def load_contextual_vector_db():
    BASE_DIR = Path(__file__).resolve().parents[1]
    with open(BASE_DIR / "answer" / "contextual_chunks.json", "r", encoding="utf-8") as f:
        contextual_chunks = json.load(f)

    vector_db = []
    for chunk in contextual_chunks:
        emb = get_embedding(chunk)
        vector_db.append((chunk, emb))

    return vector_db

VECTOR_DB_CONTEXTUAL = load_contextual_vector_db()

def retrieve(query, vector_db, top_k=3):
    query_emb = get_embedding(query)
    scored_chunks = []

    for chunk, emb in vector_db:
        score = cosine_similarity(query_emb, emb)
        scored_chunks.append((chunk, score))

    scored_chunks = sorted(scored_chunks, key=lambda x: x[1], reverse=True)
    return scored_chunks[:top_k]

def answer_question_contextual(query, vector_db, top_k=3, model_name="llama-3.1-8b-instant"):
    retrieved = retrieve(query, vector_db, top_k=top_k)
    context = "\n\n".join([chunk for chunk, _ in retrieved])

    prompt = f"""
You are answering a question using only the provided context.

Context:
{context}

Question:
{query}

Instructions:
- Answer based only on the provided context.
- If the answer is not stated word-for-word, infer it from the context when it is clearly supported.
- Keep the answer concise, around 1-3 sentences.
- Only say "The answer is not found in the provided context." if there is truly no relevant information.
"""

    response = client.chat.completions.create(
        model=model_name,
        messages=[{"role": "user", "content": prompt}],
        temperature=0
    )

    answer = response.choices[0].message.content.strip()
    return answer, retrieved

# ----------------------------
# UI
# ----------------------------
question = st.text_input("Ask a question about Chapter 8 (Transformers):")

if st.button("Ask") and question.strip():
    with st.spinner("Generating answer..."):
        try:
            answer, retrieved = answer_question_contextual(question, VECTOR_DB_CONTEXTUAL, top_k=3)

            st.subheader("Answer")
            st.write(answer)

            st.subheader("Source Chunks")
            for i, (chunk, score) in enumerate(retrieved, 1):
                with st.expander(f"Chunk {i} (score={score:.4f})"):
                    st.write(chunk)

        except Exception as e:
            st.error(f"Error: {e}")