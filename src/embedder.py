import os
import json
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

INDEX_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "index")
INDEX_PATH = os.path.join(INDEX_DIR, "faiss.index")
CHUNKS_PATH = os.path.join(INDEX_DIR, "chunks.json")

MODELE_NOM = "paraphrase-multilingual-MiniLM-L12-v2"


def charger_modele() -> SentenceTransformer:
    print(f"  [MODEL] Chargement : {MODELE_NOM}")
    return SentenceTransformer(MODELE_NOM)


def creer_index(chunks_avec_meta: list, modele: SentenceTransformer):
    print(f"  [ENCODE] Encodage de {len(chunks_avec_meta)} chunks...")

    textes = [c["contenu"] for c in chunks_avec_meta]

    vecteurs = modele.encode(
        textes,
        batch_size=64,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True,
    )

    dimension = vecteurs.shape[1]
    print(f"  [OK] Dimension des vecteurs : {dimension}")

    # IndexFlatIP + vecteurs normalises L2 = similarite cosinus
    index = faiss.IndexFlatIP(dimension)
    index.add(vecteurs.astype(np.float32))

    print(f"  [OK] Index cree avec {index.ntotal} vecteurs")
    return index


def sauvegarder_index(index, chunks_avec_meta: list, chemin_base: str = None):
    if chemin_base is None:
        chemin_base = INDEX_DIR

    os.makedirs(chemin_base, exist_ok=True)

    idx_path = os.path.join(chemin_base, "faiss.index")
    chunks_path = os.path.join(chemin_base, "chunks.json")

    faiss.write_index(index, idx_path)
    print(f"  [OK] Index FAISS sauvegarde : {idx_path}")

    with open(chunks_path, "w", encoding="utf-8") as f:
        json.dump(chunks_avec_meta, f, ensure_ascii=False, indent=2)
    print(f"  [OK] Metadonnees sauvegardees : {chunks_path}")


def charger_index(chemin_base: str = None):
    if chemin_base is None:
        chemin_base = INDEX_DIR

    idx_path = os.path.join(chemin_base, "faiss.index")
    chunks_path = os.path.join(chemin_base, "chunks.json")

    if not os.path.exists(idx_path) or not os.path.exists(chunks_path):
        raise FileNotFoundError("Index non trouve. Lance d'abord : python indexer.py")

    index = faiss.read_index(idx_path)
    with open(chunks_path, encoding="utf-8") as f:
        chunks_avec_meta = json.load(f)

    print(f"  [OK] Index charge : {index.ntotal} vecteurs, {len(chunks_avec_meta)} chunks")
    return index, chunks_avec_meta
