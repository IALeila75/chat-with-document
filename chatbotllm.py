import streamlit as st
import os
from dotenv import load_dotenv
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from langchain.text_splitter import RecursiveCharacterTextSplitter
from openai import OpenAI
from PyPDF2 import PdfReader
import re
from collections import Counter
from wordcloud import WordCloud
import matplotlib.pyplot as plt
from transformers import pipeline


import string
from wordcloud import WordCloud, STOPWORDS


import string
import spacy
from wordcloud import STOPWORDS


USE_LOCAL = os.getenv("USE_LOCAL_MODELS", "false").lower() == "true"
summarizer = gen_s2s = qa_pipeline = None

# … plus bas, dans la section Chatbot …
if USE_LOCAL:
    from transformers import pipeline
    summarizer = pipeline("summarization", ... , local_files_only=True)
    gen_s2s    = pipeline("text2text-generation", ... , local_files_only=True)
    qa_pipeline= pipeline("question-answering", ... , local_files_only=True)



try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass


# Chargez le modèle spaCy (français ici)
try:
    nlp = spacy.load("fr_core_news_sm")
except OSError:
    st.error(
        "❌ Le modèle spaCy 'fr_core_news_sm' n'est pas installé.\n"
        "Exécutez : python -m spacy download fr_core_news_sm"
    )
    st.stop()


def clean_text(text, min_word_length: int = 4) -> str:
    # 1. Passage en minuscules
    txt = text.lower()
    # 2. Suppression de la ponctuation et des chiffres
    txt = re.sub(rf"[{re.escape(string.punctuation)}0-9]", " ", txt)
    # 3. Normalisation des espaces
    txt = re.sub(r"\s+", " ", txt).strip()

    # 4. Tokenisation + filtre sur longueur + lemmatisation + suppression stopwords
    doc = nlp(txt)
    tokens = []
    for token in doc:
        if token.is_punct or token.is_space or token.is_stop:
            continue
        lemma = token.lemma_.strip()
        if len(lemma) < min_word_length:
            continue
        tokens.append(lemma)

    return " ".join(tokens)



# --- Chargement des variables d'environnement ---
load_dotenv(dotenv_path=".env")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    st.error("🔑 Veuillez définir OPENAI_API_KEY dans .env à la racine du projet.")
    st.stop()

# --- Client OpenAI v1 ---
db_client = OpenAI(api_key=OPENAI_API_KEY , timeout=30)

# --- Modèle d'embeddings multilingue ---
try:
    embed_model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')
except Exception as e:
    st.error(f"❌ Impossible de charger le modèle d'embeddings multilingue : {e}")
    st.stop()

# --- Pipelines Transformers locaux ---
try:
    summarizer = pipeline(
        "summarization",
        model="facebook/bart-large-cnn",
        tokenizer="facebook/bart-large-cnn"
    )
    gen_s2s = pipeline(
        "text2text-generation",
        model="google/flan-t5-small",
        tokenizer="google/flan-t5-small"
    )
    qa_pipeline = pipeline(
        "question-answering",
        model="distilbert-base-cased-distilled-squad",
        tokenizer="distilbert-base-cased-distilled-squad"
    )
except Exception as e:
    st.error(f"❌ Impossible d'initialiser les pipelines Transformers : {e}")
    st.stop()

# --- Dossier de stockage des PDFs ---
DATA_DIR = "data"
os.makedirs(DATA_DIR, exist_ok=True)

# --- Navigation latérale ---
st.sidebar.title("Navigation")
page = st.sidebar.radio("Aller à", ["Data Upload", "Visualisation", "Chatbot"])

# --- Helpers cachés par Streamlit ---
@st.cache_data(show_spinner=False)
def list_files(ext):
    return [f for f in os.listdir(DATA_DIR) if f.lower().endswith(ext)]

@st.cache_data(show_spinner=False)
def extract_pdf_pages(path):
    reader = PdfReader(path)
    pages = []
    for page in reader.pages:
        text = page.extract_text() or ""
        pages.append(text.replace("\n", " ").strip())
    return pages

@st.cache_resource(show_spinner=False)
def get_embeddings_and_index(path, chunks):
    emb_path = f"{path}.emb.npy"
    idx_path = f"{path}.idx"
    # Chargement du cache si existant
    if os.path.exists(emb_path) and os.path.exists(idx_path):
        embeddings = np.load(emb_path)
        index = faiss.read_index(idx_path)
        return embeddings, index
    # Calcul des embeddings
    embeddings = embed_model.encode(chunks, show_progress_bar=False)
    emb_array = np.array(embeddings, dtype=np.float32)
    # Construction de l'index FAISS
    dim = emb_array.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(emb_array)
    # Sauvegarde du cache
    np.save(emb_path, emb_array)
    faiss.write_index(index, idx_path)
    return emb_array, index

# --- Page : Data Upload ---
if page == "Data Upload":
    st.header("Charger ou sélectionner un PDF")
    old_path = st.session_state.get("pdf_path", None)

    uploaded = st.file_uploader("Envoyez un PDF", type=["pdf"])
    if uploaded:
        path = os.path.join(DATA_DIR, uploaded.name)
        with open(path, "wb") as f:
            f.write(uploaded.read())
        st.success(f"PDF sauvegardé : {uploaded.name}")
        st.session_state['pdf_path'] = path
    else:
        files = list_files(".pdf")
        if files:
            sel = st.selectbox("Ou choisissez un PDF existant", files)
            if sel:
                st.session_state['pdf_path'] = os.path.join(DATA_DIR, sel)
        else:
            st.info("Aucun PDF dans le dossier `data/`.")

    # Si on change de PDF, on vide le cache d'index
    if old_path and old_path != st.session_state.get("pdf_path"):
        for key in ['chunks', 'faiss_index']:
            if key in st.session_state:
                del st.session_state[key]

    if 'pdf_path' in st.session_state:
        st.write(f"Document sélectionné : **{os.path.basename(st.session_state['pdf_path'])}**")

# --- Page : Visualisation ---
elif page == "Visualisation":
    st.header("Visualisation du PDF")
    if 'pdf_path' in st.session_state:
        path = st.session_state['pdf_path']
        pages = extract_pdf_pages(path)

        # Statistiques globales
        st.subheader(" Statistiques")
        st.markdown(f"- **Pages** : {len(pages)}")
        if 'chunks' in st.session_state and 'faiss_index' in st.session_state:
            chunks = st.session_state['chunks']
            embeddings, _ = get_embeddings_and_index(path, chunks)
            st.markdown(f"- **Chunks** : {len(chunks)}")
            st.markdown(f"- **Vecteurs** : {embeddings.shape[0]} × {embeddings.shape[1]}")

        # Affichage de la page sélectionnée
        num = st.number_input("Afficher la page", min_value=1, max_value=len(pages), value=1)
        text = pages[num-1]
        st.text_area(f"Page {num}", text, height=300)
       
       

        # Nettoyage avancé
        cleaned = clean_text(text)

        # WordCloud sur le texte nettoyé
        st.subheader(f"️ WordCloud – Page {num}")
        wc = WordCloud(
            width=800, height=400,
            background_color="white",
            stopwords=set(STOPWORDS)
        ).generate(cleaned)
        st.image(wc.to_array(), use_column_width=True)

        # Diagramme en barres sur le texte nettoyé
        st.subheader(" Top 10 mots – Page sélectionnée")
        tokens = cleaned.split()
        freq = Counter(tokens)
        top10 = freq.most_common(10)
        if top10:
            terms, counts = zip(*top10)
            fig, ax = plt.subplots()
            ax.bar(terms, counts)
            ax.set_xticklabels(terms, rotation=45, ha="right")
            ax.set_ylabel("Fréquence")
            ax.set_title("Top 10 des mots")
            st.pyplot(fig)
        else:
            st.info("Pas assez de texte pour générer un histogramme de fréquences.")

        # Construction de l'index RAG si besoin
        if 'chunks' not in st.session_state:
            splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
            chunks = splitter.split_text("\n".join(pages))
            st.session_state['chunks'] = chunks
            with st.spinner(" Construction de l'index RAG local…"):
                embeddings, index = get_embeddings_and_index(path, chunks)
            st.session_state['faiss_index'] = index
            st.success(f"Index RAG prêt ({len(chunks)} chunks)")
            st.info(f"Embeddings → {path}.emb.npy")
            st.info(f"Index FAISS → {path}.idx")
    else:
        st.info("Chargez d'abord un PDF sur la page « Data Upload ».")

# --- Page : Chatbot ---
elif page == "Chatbot":
    st.header("Chat avec votre PDF")
    if 'faiss_index' in st.session_state and 'chunks' in st.session_state:
        query = st.text_input("Posez une question ou tapez 'résumé' pour un résumé :")
        if query:
            # Embedding de la requête
            q_emb = np.array(
                embed_model.encode([query], show_progress_bar=False),
                dtype=np.float32
            ).reshape(1, -1)
            D, I = st.session_state['faiss_index'].search(q_emb, k=5)
            context = "\n\n".join(st.session_state['chunks'][i] for i in I[0])

            # Si on demande un résumé
            if query.strip().lower().startswith("résumé"):
                st.info(" Génération du résumé en local…")
                try:
                    summary = summarizer(
                        context,
                        max_length=150,
                        min_length=30,
                        do_sample=False,
                        truncation=True
                    )[0]["summary_text"].strip()
                    st.text_area("Résumé (local)", summary, height=200)
                except Exception as e:
                    st.error(f"❌ Erreur résumé local : {e}")

            # Sinon, génération locale de réponse
            else:
                st.info(" Génération de réponse locale…")
                try:
                    gen_input = (
                        "Contexte :\n" + context +
                        "\n\nQuestion : " + query +
                        "\nRéponse :"
                    )
                    generated = gen_s2s(
                        gen_input,
                        max_length=150,
                        do_sample=False
                    )[0]["generated_text"].strip()
                    st.text_area("Réponse (local génératif)", generated, height=200)
                except Exception as e:
                    st.error(f"❌ Erreur génération locale : {e}")
    else:
        st.info("Construisez d'abord l'index local sur la page « Visualisation ».")

# --- Footer ---
st.sidebar.markdown("---")
st.sidebar.markdown("*Streamlit PDF Chatbot & Viz Demo*")
