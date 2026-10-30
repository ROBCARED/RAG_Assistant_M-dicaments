# 💊 Assistant Médicaments — RAG sur la BDPM

Assistant conversationnel qui répond aux questions sur les médicaments vendus en France (indications, posologie, effets indésirables, contre-indications, interactions, grossesse…) à partir des données officielles de la **Base de Données Publique des Médicaments (BDPM)**, et qui **cite ses sources**.

> ⚠️ Projet pédagogique : les réponses ne remplacent pas l'avis d'un médecin ou d'un pharmacien.

## Fonctionnement

```
BDPM (fichiers officiels + RCP)  →  découpage par section  →  embeddings  →  index FAISS
                                                                              │
Question  →  reformulation (LLM)  →  recherche vectorielle  →  reranking  →  réponse sourcée (LLM)
```

1. **Collecte** : téléchargement des fichiers de la BDPM et scraping des RCP (résumés des caractéristiques du produit) avec Requests et BeautifulSoup.
2. **Découpage** : chaque médicament est découpé par section (indications, posologie, effets indésirables, grossesse, contre-indications, interactions…).
3. **Indexation** : embeddings multilingues (`paraphrase-multilingual-MiniLM-L12-v2`) stockés dans un index **FAISS**.
4. **Recherche** : un agent *Retriever* reformule la question en tenant compte de l'historique, récupère les 5 passages les plus proches, les réordonne avec un cross-encoder et vérifie un seuil de confiance.
5. **Génération** : un agent *Generator* rédige la réponse avec **Llama 3.3 70B** (API Groq), uniquement à partir des passages retrouvés.

## Stack

Python · sentence-transformers · FAISS · Groq (Llama 3.3 70B) · Streamlit · BeautifulSoup

## Lancer le projet

```bash
pip install -r requirements.txt
echo "GROQ_API_KEY=votre_cle" > .env

python indexer.py        # télécharge les données et construit l'index FAISS
streamlit run app.py     # interface web
# ou : python rag.py     # version en ligne de commande
```

## Organisation du code

| Fichier | Rôle |
|---|---|
| `src/loader.py`, `src/rcp_scraper.py` | Téléchargement et scraping des données BDPM |
| `src/chunker.py` | Découpage des fiches par section |
| `src/embedder.py` | Modèle d'embedding et index FAISS |
| `src/agents.py` | Agents de recherche (reformulation, reranking) et de génération |
| `indexer.py` | Construction de l'index |
| `app.py` / `rag.py` | Interface Streamlit / ligne de commande |
