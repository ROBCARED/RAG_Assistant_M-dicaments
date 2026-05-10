import numpy as np
from sentence_transformers import SentenceTransformer

SEUIL_CONFIANCE = 0.35


def rechercher(
    question: str,
    modele: SentenceTransformer,
    index,
    chunks_avec_meta: list[dict],
    k: int = 4,
) -> list[dict]:
    vecteur = modele.encode(
        [question],
        convert_to_numpy=True,
        normalize_embeddings=True,
    ).astype(np.float32)

    scores, indices = index.search(vecteur, k)

    resultats = []
    for score, idx in zip(scores[0], indices[0]):
        if idx == -1:
            continue
        resultats.append({
            "contenu": chunks_avec_meta[idx]["contenu"],
            "metadata": chunks_avec_meta[idx]["metadata"],
            "score": float(score),
        })

    return resultats


def score_est_suffisant(resultats: list[dict]) -> bool:
    if not resultats:
        return False
    return resultats[0]["score"] >= SEUIL_CONFIANCE
