"""
retriever.py — Étape 5 : Recherche vectorielle

Encode une question utilisateur, normalise le vecteur et interroge
l'index FAISS pour retourner les k chunks les plus similaires.
"""

import numpy as np
from sentence_transformers import SentenceTransformer


# Seuil de confiance : en dessous, on avertit l'utilisateur (Bonus B)
SEUIL_CONFIANCE = 0.35


def rechercher(
    question: str,
    modele: SentenceTransformer,
    index,
    chunks_avec_meta: list[dict],
    k: int = 4,
) -> list[dict]:
    """
    Recherche les k chunks les plus pertinents pour une question.

    Étapes :
    1. Encodage de la question en vecteur dense
    2. Normalisation L2 (pour obtenir une similarité cosinus avec IndexFlatIP)
    3. Recherche des k plus proches voisins dans FAISS
    4. Retour des chunks avec leur score

    Args:
        question: question de l'utilisateur en langage naturel
        modele: même modèle sentence-transformers que celui utilisé à l'indexation
        index: index FAISS chargé
        chunks_avec_meta: liste des chunks dans le même ordre que les vecteurs
        k: nombre de résultats à retourner

    Returns:
        Liste de dicts {"contenu": str, "metadata": dict, "score": float}
        triée par score décroissant (meilleur en premier)
    """
    # Encodage + normalisation (identique à ce qui a été fait à l'indexation)
    vecteur = modele.encode(
        [question],
        convert_to_numpy=True,
        normalize_embeddings=True,
    ).astype(np.float32)

    # Recherche dans FAISS
    scores, indices = index.search(vecteur, k)

    resultats = []
    for score, idx in zip(scores[0], indices[0]):
        if idx == -1:
            continue  # FAISS retourne -1 si pas assez de vecteurs
        resultats.append({
            "contenu": chunks_avec_meta[idx]["contenu"],
            "metadata": chunks_avec_meta[idx]["metadata"],
            "score": float(score),
        })

    return resultats


def score_est_suffisant(resultats: list[dict]) -> bool:
    """
    Bonus B — Vérifie si le meilleur score dépasse le seuil de confiance.

    Args:
        resultats: liste retournée par rechercher()

    Returns:
        True si des résultats pertinents ont été trouvés, False sinon
    """
    if not resultats:
        return False
    return resultats[0]["score"] >= SEUIL_CONFIANCE
