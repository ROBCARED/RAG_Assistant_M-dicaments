import os
from groq import Groq
from dotenv import load_dotenv

load_dotenv()

client = Groq(api_key=os.getenv("GROQ_API_KEY"))
MODELE_LLM = "llama-3.3-70b-versatile"


def construire_prompt_systeme() -> str:
    return """Tu es un assistant pharmaceutique expert base sur la Base de Donnees Publique des Medicaments (BDPM) francaise.

Ton role est d'aider les utilisateurs a obtenir des informations factuelles sur les medicaments : composition, forme, voies d'administration, et titulaire.

REGLES STRICTES :
1. Tu reponds UNIQUEMENT en te basant sur le contexte fourni (extraits de la BDPM).
2. Si l'information demandee n'est PAS dans le contexte, reponds : "Je n'ai pas trouve d'information directement pertinente dans ma base de donnees sur ce sujet."
3. Tu cites toujours le nom exact du medicament dans ta reponse.
4. Tu rappelles systematiquement : "Ces informations sont a titre informatif uniquement. Consultez toujours un professionnel de sante ou votre pharmacien avant toute prise de medicament."
5. Tu ne donnes PAS de conseils medicaux personnalises ni de diagnostic.
6. Tu reponds en francais.

FORMAT DE REPONSE :
- Reponse claire et structuree
- Sources citees a la fin : Sources : [Nom medicament] (BDPM)
- Avertissement medical en fin de message"""


def construire_contexte(chunks_pertinents: list[dict]) -> str:
    parties = []
    for i, chunk in enumerate(chunks_pertinents, 1):
        nom = chunk["metadata"].get("denomination", "Medicament inconnu")
        score = chunk["score"]
        parties.append(f"[Extrait {i} — {nom} (score: {score:.2f})]\n{chunk['contenu']}")
    return "\n\n".join(parties)


def generer_reponse(question: str, chunks_pertinents: list[dict]) -> str:
    contexte = construire_contexte(chunks_pertinents)

    prompt_utilisateur = (
        "Voici des extraits de la Base de Donnees Publique des Medicaments (BDPM) :\n\n"
        f"--- CONTEXTE ---\n{contexte}\n--- FIN DU CONTEXTE ---\n\n"
        f"Question : {question}\n\n"
        "Reponds en te basant UNIQUEMENT sur le contexte fourni ci-dessus."
    )

    reponse = client.chat.completions.create(
        model=MODELE_LLM,
        messages=[
            {"role": "system", "content": construire_prompt_systeme()},
            {"role": "user", "content": prompt_utilisateur},
        ],
        temperature=0.2,
        max_tokens=1024,
    )

    return reponse.choices[0].message.content


def reformuler_question(question: str) -> str:
    prompt = (
        "Tu es un expert en recherche pharmaceutique.\n"
        "Reformule la question suivante en termes pharmaceutiques precis pour optimiser "
        "une recherche dans une base de donnees de medicaments.\n"
        "Retourne UNIQUEMENT la question reformulee, sans explication.\n\n"
        f"Question originale : {question}\n"
        "Question reformulee :"
    )

    reponse = client.chat.completions.create(
        model=MODELE_LLM,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.1,
        max_tokens=150,
    )

    return reponse.choices[0].message.content.strip()
