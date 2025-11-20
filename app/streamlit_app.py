# paste full code here

# Install streamlit
import streamlit as st
import os
import pandas as pd
import numpy as np
import torch
import faiss
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch.nn.functional as F

# ------------------
# User-editable paths
# ------------------
# NOTE: These files were uploaded earlier in the environment. If you run the app elsewhere,
# update these paths or place the files in the same locations.
TRUE_CSV_PATH = "/content/True.csv"
FAKE_CSV_PATH = "/content/Fake.csv"
SAVED_MODEL_DIR = "./saved_roberta_fake_detector"  # where fine-tuned model will be saved/loaded

# ------------------
# Load data & build FAISS index (cached)
# ------------------
@st.cache_resource
def load_datasets(true_path, fake_path):
    true_df = pd.read_csv(true_path)
    fake_df = pd.read_csv(fake_path)
    # Basic cleaning (same as notebook)
    def clean_text(s):
        s = str(s)
        s = s.replace('\n',' ').replace('\r',' ')
        return " ".join(s.split())
    true_df['text'] = true_df['text'].apply(clean_text)
    fake_df['text'] = fake_df['text'].apply(clean_text)
    return true_df, fake_df

@st.cache_resource
def build_faiss_index(real_texts, model_name='all-MiniLM-L6-v2'):
    embed_model = SentenceTransformer(model_name)
    embeddings = embed_model.encode(real_texts, convert_to_numpy=True, show_progress_bar=False)
    embeddings = embeddings.astype('float32')
    dim = embeddings.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(embeddings)
    return index, embed_model, embeddings

# ------------------
# Load classification model & tokenizer
# ------------------
@st.cache_resource
def load_classifier(model_dir):
    # If a fine-tuned model is available, load it; otherwise load roberta-base
    if os.path.isdir(model_dir) and os.path.exists(os.path.join(model_dir, 'config.json')):
        tokenizer = AutoTokenizer.from_pretrained(model_dir)
        model = AutoModelForSequenceClassification.from_pretrained(model_dir)
    else:
        tokenizer = AutoTokenizer.from_pretrained('roberta-base')
        model = AutoModelForSequenceClassification.from_pretrained('roberta-base', num_labels=2)
    model.eval()
    return tokenizer, model

# ------------------
# Predict & explain
# ------------------

def classify_text(text, tokenizer, model, device='cpu'):
    inputs = tokenizer(text, return_tensors='pt', truncation=True, max_length=256)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        probs = F.softmax(logits, dim=-1).cpu().numpy()[0]
        pred = int(np.argmax(probs))
    return pred, probs


def retrieve_evidence(query, index, embed_model, real_texts, k=3):
    q_emb = embed_model.encode([query], convert_to_numpy=True).astype('float32')
    distances, indices = index.search(q_emb, k)
    hits = []
    for idx in indices[0]:
        hits.append({'text': real_texts[idx], 'idx': int(idx)})
    return hits

# ------------------
# Streamlit UI
# ------------------
st.set_page_config(page_title='Fake News Detector (RoBERTa + RAG)', layout='wide')
st.title('📰 Fake News Detector — RoBERTa + RAG')

st.sidebar.header('Settings')
show_raw = st.sidebar.checkbox('Show raw model scores', value=False)
k = st.sidebar.number_input('Number of evidence passages (k)', min_value=1, max_value=10, value=3)

# Load data and models
with st.spinner('Loading datasets and models (cached)...'):
    try:
        true_df, fake_df = load_datasets(TRUE_CSV_PATH, FAKE_CSV_PATH)
    except Exception as e:
        st.error(f"Failed to load dataset files from paths. Update paths in the app.\nError: {e}")
        st.stop()

    real_texts = true_df['text'].tolist()
    index, embed_model, embeddings = build_faiss_index(real_texts)
    tokenizer, clf_model = load_classifier(SAVED_MODEL_DIR)

# Main input area
st.subheader('Enter the news article (title + body) you want to check')
user_text = st.text_area('Paste the news text here', height=200)

col1, col2 = st.columns([2,1])
with col2:
    if st.button('Predict & Explain'):
        if not user_text.strip():
            st.warning('Please input some text to analyze.')
        else:
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
            clf_model.to(device)
            with st.spinner('Running classification...'):
                pred, probs = classify_text(user_text, tokenizer, clf_model, device=device)
                label = 'REAL' if pred==1 else 'FAKE'

            score = float(probs[pred])
            st.markdown(f"### Prediction: **{label}**  ")
            st.write(f"Confidence: {score:.3f}")
            if show_raw:
                st.write({ 'prob_fake': float(probs[0]), 'prob_real': float(probs[1]) })

            # Retrieve evidence (from real news corpus)
            with st.spinner('Retrieving evidence passages...'):
                evidence = retrieve_evidence(user_text, index, embed_model, real_texts, k=k)

            st.markdown('---')
            st.markdown('### Retrieved evidence (from REAL news corpus)')
            for i, ev in enumerate(evidence, start=1):
                st.markdown(f"**Evidence {i}** (doc id: {ev['idx']})")
                st.write(ev['text'])

            st.markdown('---')
            st.info('Note: This system predicts based on patterns in the provided dataset and retrieves similar real-news passages as evidence. For real-world deployment, add more datasets and fact-checking sources.')

# Footer
st.markdown('---')
st.caption('App generated by your assistant — edit dataset paths or model path at the top of the file if needed.')
