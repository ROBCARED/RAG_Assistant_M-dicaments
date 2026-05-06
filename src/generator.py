"""
generator.py — Étape 6 : Génération de réponse avec Groq

Construit le prompt, appelle l'API Groq et retourne la réponse du LLM.
Implémente également le Bonus C (reformulation de question).
"""

import os
from groq import Groq
from dotenv import load_dotenv

load_dotenv()

# Client Groq
client = Groq(api_key=os.getenv("GROQ_API_KEY"))

# Modèle utilisé
MODELE_LLM = "llama3-70b-8192"


def construire_prompt_systeme() -> str:
    """
    Retourne le prompt système pour l'assistant médicaments.

    Définit :
    - Le rôle : assistant pharmaceutique basé sur la BDPM
    - Les contraintes : ne pas inventer, citer les sources, avertissements médicaux
    - Le format : réponse structurée avec sources en fin de message
    """
    return """Tu es un assistant pharmaceutique expert basé sur la Base de Données Publique des Médicaments (BDPM) française.

Ton rôle est d'aider les utilisateurs à obtenir des informations factuelles sur les médicaments : composition, forme, voies d'administration, et titulaire.

RÈGLES STRICTES :
1. Tu réponds UNIQUEMENT en te basant sur le contexte fourni (extraits de la BDPM).
2. Si l'information demandée n'est PAS dans le contexte, tu réponds exactement : "Je n'ai pas trouvé d'information directement pertinente dans ma base de données sur ce sujet."
3. Tu cites toujours le nom exact du médicament (BDPM) dans ta réponse.
4. Tu rappelles systématiquement : "⚠️ Ces informations sont à titre informatif uniquement. Consultez toujours un professionnel de santé ou votre pharmacien avant toute prise de médicament."
5. Tu ne donnes PAS de conseils médicaux personnalisés ni de diagnostic.
6. Tu répondis en français.

FORMAT DE RÉPONSE :
- Réponse claire et structurée
- Sources citées à la fin sous la forme : 📋 Sources : [Nom médicament] (BDPM)
- Avertissement médical en fin de message"""


def construire_contexte(chunks_pertinents: list[dict]) -> str:
    """
    Assemble les chunks en un texte de contexte pour le LLM.

    Args:
        chunks_pertinents: résultats retournés par rechercher()

    Returns:
        Texte de contexte formaté pour inclusion dans le prompt
    """
    parties = []
    for i, chunk in enumerate(chunks_pertinents, 1):
        nom = chunk["metadata"].get("denomination", "Médicament inconnu")
        score = chunk["score"]
        parties.append(f"[Extrait {i} — {nom} (score: {score:.2f})]\n{chunk['contenu']}")

    return "\n\n".join(parties)


def generer_reponse(question: str, chunks_pertinents: list[dict]) -> str:
    """
    Génère une réponse en utilisant les chunks comme contexte.

    Args:
        question: la question de l'utilisateur
        chunks_pertinents: les résultats de la recherche vectorielle

    Returns:
        La réponse générée par le LLM Groq
    """
    contexte = construire_contexte(chunks_pertinents)

    prompt_utilisateur = f"""Voici des extraits de la Base de Données Publique des Médicaments (BDPM) pertinents pour répondre à la question :

--- CONTEXTE ---
{contexte}
--- FIN DU CONTEXTE ---

Question de l'utilisateur : {question}

Réponds en te basant UNIQUEMENT sur le contexte fourni ci-dessus."""

    reponse = client.chat.completions.create(
        model=MODELE_LLM,
        messages=[
            {"role": "system", "content": construire_prompt_systeme()},
            {"role": "user", "content": prompt_utilisateur},
        ],
        temperature=0.2,  # Faible pour des réponses factuelles cohérentes
        max_tokens=1024,
    )

    return reponse.choices[0].message.content


def reformuler_question(question: str) -> str:
    """
    Bonus C — Reformule la question pour améliorer la recherche vectorielle.

    Transforme une question en langage naturel en termes de recherche
    pharmaceutique plus précis (noms de substances, formes, voies...).

    Args:
        question: question originale de l'utilisateur

    Returns:
        Question reformulée optimisée pour la recherche vectorielle
    """
    prompt = f"""Tu es un expert en recherche pharmaceutique. 
Reformule la question suivante en termes pharmaceutiques précis pour optimiser 
une recherche dans une base de données de médicaments.
Retourne UNIQUEMENT la question reformulée, sans explication.

Question originale : {question}
Question reformulée :"""

    reponse = client.chat.completions.create(
        model="llama3-8b-8192",  # Modèle rapide pour cette tâche simple
        messages=[{"role": "user", "content": prompt}],
        temperature=0.1,
        max_tokens=150,
    )

    reformulee = reponse.choices[0].message.content.strip()
    return reformulee
