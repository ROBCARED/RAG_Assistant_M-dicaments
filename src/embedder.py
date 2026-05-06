"""
embedder.py - Etape 3 : Creation des embeddings et index FAISS

Encode les chunks en vecteurs denses avec sentence-transformers,
construit un index FAISS IndexFlatIP (similarite cosinus apres normalisation)
et le persiste sur disque.
"""

import os
import json
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

# Repertoires
INDEX_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "index")
INDEX_PATH = os.path.join(INDEX_DIR, "faiss.index")
CHUNKS_PATH = os.path.join(INDEX_DIR, "chunks.json")

# Modele multilingue (supporte le francais)
MODELE_NOM = "paraphrase-multilingual-MiniLM-L12-v2"


def charger_modele() -> SentenceTransformer:
    """Charge le modele d'embedding (telecharge automatiquement si absent)."""
    print(f"  [MODEL] Chargement : {MODELE_NOM}")
    return SentenceTransformer(MODELE_NOM)


def creer_index(chunks_avec_meta: list, modele: SentenceTransformer):
    """
    Encode tous les chunks et construit l'index FAISS.

    Utilise IndexFlatIP avec des vecteurs normalises L2,
    ce qui revient a calculer la similarite cosinus.

    Args:
        chunks_avec_meta: liste de dicts {"contenu": str, "metadata": dict}
        modele: modele sentence-transformers charge

    Returns:
        Index FAISS rempli
    """
    print(f"  [ENCODE] Encodage de {len(chunks_avec_meta)} chunks...")

    textes = [c["contenu"] for c in chunks_avec_meta]

    # Encodage par batch avec barre de progression
    vecteurs = modele.encode(
        textes,
        batch_size=64,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True,  # Normalisation L2 pour similarite cosinus
    )

    dimension = vecteurs.shape[1]
    print(f"  [OK] Dimension des vecteurs : {dimension}")

    # IndexFlatIP = produit scalaire = cosinus si vecteurs normalises
    index = faiss.IndexFlatIP(dimension)
    index.add(vecteurs.astype(np.float32))

    print(f"  [OK] Index cree avec {index.ntotal} vecteurs")
    return index


def sauvegarder_index(index, chunks_avec_meta: list, chemin_base: str = None):
    """
    Sauvegarde l'index FAISS et les metadonnees associees sur disque.

    L'index FAISS associe chaque vecteur a son rang (entier).
    On sauvegarde la liste des chunks dans le meme ordre pour
    retrouver le texte correspondant a un resultat de recherche.

    Args:
        index: index FAISS a sauvegarder
        chunks_avec_meta: liste des chunks (meme ordre que les vecteurs)
        chemin_base: repertoire de sauvegarde (par defaut : INDEX_DIR)
    """
    if chemin_base is None:
        chemin_base = INDEX_DIR

    os.makedirs(chemin_base, exist_ok=True)

    idx_path = os.path.join(chemin_base, "faiss.index")
    chunks_path = os.path.join(chemin_base, "chunks.json")

    # Sauvegarde de l'index FAISS
    faiss.write_index(index, idx_path)
    print(f"  [OK] Index FAISS sauvegarde : {idx_path}")

    # Sauvegarde des metadonnees
    with open(chunks_path, "w", encoding="utf-8") as f:
        json.dump(chunks_avec_meta, f, ensure_ascii=False, indent=2)
    print(f"  [OK] Metadonnees sauvegardees : {chunks_path}")


def charger_index(chemin_base: str = None):
    """
    Recharge l'index FAISS et les chunks depuis le disque sans reindexer.

    Args:
        chemin_base: repertoire ou sont stockes les fichiers

    Returns:
        Tuple (index FAISS, liste des chunks)

    Raises:
        FileNotFoundError: si l'index n'a pas encore ete cree
    """
    if chemin_base is None:
        chemin_base = INDEX_DIR

    idx_path = os.path.join(chemin_base, "faiss.index")
    chunks_path = os.path.join(chemin_base, "chunks.json")

    if not os.path.exists(idx_path) or not os.path.exists(chunks_path):
        raise FileNotFoundError(
            "Index non trouve. Lance d'abord : python indexer.py"
        )

    index = faiss.read_index(idx_path)
    with open(chunks_path, encoding="utf-8") as f:
        chunks_avec_meta = json.load(f)

    print(f"  [OK] Index charge : {index.ntotal} vecteurs, {len(chunks_avec_meta)} chunks")
    return index, chunks_avec_meta
