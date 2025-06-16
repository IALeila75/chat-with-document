# Chat-with-Document : Chatbot PDF & Visualisation

Une application Streamlit qui vous permet de :
-  **Dialoguer** avec un document PDF volumineux (question-réponse & résumé)  
-  **Explorer** le contenu page par page  
-  **Visualiser** la répartition lexicale via Word Cloud et histogramme  
-  **Indexer** et rechercher le contexte grâce à RAG (Retrieval-Augmented Generation)  
-  **Fonctionner 100 % en local** ou basculer sur OpenAI selon quota  

---

## Fonctionnalités

1. **Data Upload**  
   - Import de PDF depuis l’interface ou sélection dans `data/`.  
   - Nettoyage et extraction du texte (PyPDF2).  

2. **Visualisation**  
   - Nombre de pages, chunks et vecteurs.  
   - Affichage du contenu page par page.  
   - Word Cloud & top-10 mots (nettoyés, stop-words, lemmatisation).  

3. **Chatbot**  
   - Recherche RAG : chunking → embeddings multilingues → FAISS.  
   - Génération OpenAI (GPT-3.5) ou fallback local (BART / Flan-T5).  
   - Résumé, Q&A ou réponse générative.  

---

## Installation

1. **Cloner le repo**  
   ```bash
   git clone https://github.com/IALeila75/chat-with-document.git
   cd chat-with-document
