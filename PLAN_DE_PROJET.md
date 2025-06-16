
---

**PLAN_DE_PROJET.md**  
```markdown
# Plan de Projet : Chat-with-Document

## 1. Contexte & Objectifs

- **Contexte** : besoins d’explorer et d’interroger de gros PDF (rapports, manuels, thèses) sans tout lire.  
- **Objectif** : fournir une interface web simple pour visualiser, rechercher et résumer ces documents via RAG + LLM et fallback local.

---

## 2. Périmètre

- **Inclus** :  
  - Upload / sélection de PDF  
  - Extraction & nettoyage du texte  
  - Chunking + embeddings + index FAISS  
  - Visualisation (textuelle, Word Cloud, bar chart)  
  - Chatbot (OpenAI et fallback local)  
  - Déploiement sur Streamlit Cloud  

- **Exclus** :  
  - OCR pour PDF scannés  
  - Authentification utilisateur  
  - Edition collaborative en temps réel  

---

## 3. Technologies & Outils

| Fonctionnalité         | Librairie / Service                       |
 |------------------------|-------------------------------------------|
| Serveur web            | Streamlit                                 |
| Extraction PDF         | PyPDF2                                    |
| Nettoyage & Lemmatisation | spaCy + regex                           |
| Chunking               | LangChain `RecursiveCharacterTextSplitter`|
| Embeddings             | SentenceTransformers (multilingue)        |
| Indexation             | FAISS                                     |
| LLM / API              | OpenAI GPT-3.5 / fallback Transformers    |
| Visualisations         | WordCloud, Matplotlib                     |
| Déploiement            | Streamlit Community Cloud                 |

---

## 4. Ressources & Responsabilités

- **Développeur principal** : configuration, code RAG, UI Streamlit  
- **Data engineer** : prétraitement PDF, nettoyage, embeddings  
- **ML engineer** : pipelines Transformers, tests de performance  
- **DevOps** : CI/CD, déploiement Streamlit Cloud, monitoring  

---

## 5. Livrables

1. **Code source** complet sur GitHub  
2. **README.md** clair pour l’installation et l’usage  
3. **Application déployée** sur Streamlit Cloud  
4. **Rapport de tests** et **guide de maintenance**  

---

## 6. Risques & Mitigations

| Risque                         | Impact            | Plan de mitigation                           |
|--------------------------------|-------------------|-----------------------------------------------|
| Quota OpenAI insuffisant       | Blocage chatbot   | Implémenter fallback local (transformers)     |
| PDF scanné (image)             | Lecture échouée   | Ajouter OCR (Tesseract) en phase ultérieure   |
| Latence embeddings / FAISS     | Mauvaise UX       | Caching disque & batch embeddings             |
| Conflits de versions libs      | Erreurs runtime   | Verrouiller versions dans requirements.txt    |

---
