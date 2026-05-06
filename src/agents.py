"""
agents.py - Architecture multi-agents

Implemente deux agents specialises qui collaborent pour repondre
aux questions sur les medicaments :

  Agent 1 - RetrieverAgent :
    - Recoit la question de l'utilisateur
    - Reformule la question pour optimiser la recherche (Bonus C)
    - Interroge l'index FAISS et retourne les chunks pertinents
    - Evalue la confiance des resultats (Bonus B)

  Agent 2 - GeneratorAgent :
    - Recoit la question + les chunks du RetrieverAgent
    - Construit un prompt contextuel adapte au type de chunk
    - Appelle l'API Groq et retourne la reponse

Flux de traitement :
  Question -> RetrieverAgent -> {chunks + metadonnees} -> GeneratorAgent -> Reponse
"""

import os
import numpy as np
from groq import Groq
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer

load_dotenv()

# ─────────────────────────────────────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────────────────────────────────────

SEUIL_CONFIANCE = 0.35      # Score minimum pour considerer un resultat pertinent
K_RESULTATS = 5             # Nombre de chunks retournes par la recherche
MODELE_LLM = "llama-3.3-70b-versatile"
MODELE_LLM_RAPIDE = "llama-3.3-70b-versatile"


# ─────────────────────────────────────────────────────────────────────────────
# Agent 1 : RetrieverAgent
# ─────────────────────────────────────────────────────────────────────────────

class RetrieverAgent:
    """
    Agent 1 : Recherche et recuperation des chunks pertinents.

    Responsabilites :
    - Reformuler la question utilisateur en termes pharmaceutiques (Bonus C)
    - Encoder la question en vecteur dense
    - Interroger l'index FAISS
    - Evaluer le score de confiance des resultats (Bonus B)
    - Retourner les chunks classes par pertinence
    """

    def __init__(self, modele: SentenceTransformer, index, chunks: list, avec_reformulation: bool = True):
        """
        Args:
            modele: modele sentence-transformers (meme que celui utilise a l'indexation)
            index: index FAISS charge
            chunks: liste des chunks dans le meme ordre que les vecteurs FAISS
            avec_reformulation: active le Bonus C (reformulation de question)
        """
        self.modele = modele
        self.index = index
        self.chunks = chunks
        self.avec_reformulation = avec_reformulation
        self.client_groq = Groq(api_key=os.getenv("GROQ_API_KEY"))

    def reformuler_question(self, question: str, historique: list = None) -> str:
        """
        Bonus C - Reformule la question en termes pharmaceutiques precis.
        Bonus A - Utilise l'historique pour deduire le contexte (ex: "et pour les femmes enceintes ?").

        Transforme "j'ai mal a la tete, que prendre ?" en
        "medicaments antidouleur cephalees analgesiques posologie".

        Args:
            question: question originale de l'utilisateur

        Returns:
            Question reformulee (ou question originale en cas d'erreur)
        """
        contexte_historique = ""
        if historique:
            contexte_historique = "Historique recent de la conversation :\n"
            for msg in historique[-4:]:  # 2 derniers echanges max
                role = "Utilisateur" if msg["role"] == "user" else "Assistant"
                # On tronque la reponse de l'assistant pour eviter de surcharger le prompt
                contenu = msg['content'][:300] + "..." if len(msg['content']) > 300 else msg['content']
                contexte_historique += f"- {role}: {contenu}\n"
            contexte_historique += "\n"

        prompt = (
            "Tu es un expert en pharmacologie. L'utilisateur va te decrire des symptomes ou poser une question.\n"
            f"{contexte_historique}"
            "En tenant compte de l'historique si necessaire (pour comprendre de quel medicament ou sujet on parle), "
            "reformule la NOUVELLE demande en termes pharmaceutiques et medicaux precis (indications therapeutiques, "
            "symptomes cliniques) pour optimiser une recherche vectorielle dans la Base de Donnees Publique des Medicaments (BDPM).\n"
            "Retourne UNIQUEMENT la question reformulee, sans explication.\n\n"
            f"Nouvelle Question/Symptomes : {question}\n"
            "Recherche reformulee :"
        )
        try:
            reponse = self.client_groq.chat.completions.create(
                model=MODELE_LLM_RAPIDE,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.1,
                max_tokens=150,
            )
            return reponse.choices[0].message.content.strip()
        except Exception:
            return question  # Repli sur la question originale

    def rechercher(self, question: str, k: int = K_RESULTATS) -> list:
        """
        Encode la question et interroge l'index FAISS.

        Args:
            question: question a encoder
            k: nombre de resultats a retourner

        Returns:
            Liste de dicts {"contenu": str, "metadata": dict, "score": float}
        """
        vecteur = self.modele.encode(
            [question],
            convert_to_numpy=True,
            normalize_embeddings=True,
        ).astype(np.float32)

        scores, indices = self.index.search(vecteur, k)

        resultats = []
        for score, idx in zip(scores[0], indices[0]):
            if idx == -1:
                continue
            resultats.append({
                "contenu": self.chunks[idx]["contenu"],
                "metadata": self.chunks[idx]["metadata"],
                "score": float(score),
            })
        return resultats

    def confiance_suffisante(self, resultats: list) -> bool:
        """
        Bonus B - Verifie si le meilleur score depasse le seuil de confiance.

        Un score faible indique que la BDPM ne contient pas
        d'information pertinente pour cette question.

        Args:
            resultats: liste retournee par rechercher()

        Returns:
            True si le resultat le plus proche est suffisamment pertinent
        """
        if not resultats:
            return False
        return resultats[0]["score"] >= SEUIL_CONFIANCE

    def run(self, question: str, historique: list = None) -> dict:
        """
        Execution complete de l'Agent 1.

        Etapes :
        1. Reformulation de la question avec historique (Bonus A & C)
        2. Recherche vectorielle
        3. Evaluation de la confiance (Bonus B)

        Args:
            question: question originale de l'utilisateur

        Returns:
            Dict avec :
            - "question_originale" : question telle que posee
            - "question_recherche" : question (eventuellemnt reformulee)
            - "reformulee" : True si reformulation effectuee
            - "resultats" : liste de chunks avec scores
            - "confiance_ok" : True si le score depasse le seuil
            - "meilleur_score" : score du chunk le plus proche
        """
        question_recherche = question
        reformulee = False

        if self.avec_reformulation:
            question_reformulee = self.reformuler_question(question, historique)
            if question_reformulee and question_reformulee != question:
                question_recherche = question_reformulee
                reformulee = True

        resultats = self.rechercher(question_recherche)
        confiance_ok = self.confiance_suffisante(resultats)
        meilleur_score = resultats[0]["score"] if resultats else 0.0

        return {
            "question_originale": question,
            "question_recherche": question_recherche,
            "reformulee": reformulee,
            "resultats": resultats,
            "confiance_ok": confiance_ok,
            "meilleur_score": meilleur_score,
        }


# ─────────────────────────────────────────────────────────────────────────────
# Agent 2 : GeneratorAgent
# ─────────────────────────────────────────────────────────────────────────────

class GeneratorAgent:
    """
    Agent 2 : Generation de la reponse via Groq.

    Responsabilites :
    - Construire un prompt contextuel a partir des chunks
    - Adapter le ton selon le type de chunks recuperes
    - Appeler l'API Groq (llama3-70b-8192)
    - Inclure les avertissements medicaux obligatoires
    - Citer les sources (noms des medicaments)
    """

    PROMPT_SYSTEME = """Tu es un Systeme d'Aide a la Decision Medicale (SADM) expert base sur la Base de Donnees Publique des Medicaments (BDPM) francaise.

Ton role est d'accompagner les professionnels de sante dans leurs prises de decision en fournissant des informations cliniques factuelles (indications therapeutiques, conditions de prescription, contre-indications, grossesse).

REGLES STRICTES :
1. Reponds UNIQUEMENT en te basant sur le contexte clinique fourni (extraits BDPM).
2. Si l'utilisateur decrit des symptomes ou une situation clinique, identifie le ou les medicaments du contexte dont les indications therapeutiques correspondent, et cite precisement ces indications.
3. Si l'information clinique n'est PAS dans le contexte, reponds exactement : "Je n'ai pas trouve d'information directement pertinente dans ma base de donnees pour eclairer cette decision."
4. Cite toujours le nom exact du medicament dans ta reponse et la section du RCP correspondante si applicable.
5. Assure un ton clinique, professionnel et objectif.
6. Termine systematiquement par : "[!] Cet outil d'aide a la decision est fourni a titre informatif. Il ne se substitue en aucun cas a l'evaluation clinique et a la responsabilite du prescripteur."
7. Reponds en francais.

FORMAT :
- Reponse claire, synthetique et structuree par points (ex: Indications, Prescription, Contre-indications).
- Sources en fin de message : [Sources : Nom medicament (BDPM)]"""

    def __init__(self):
        self.client = Groq(api_key=os.getenv("GROQ_API_KEY"))

    def _construire_contexte(self, chunks: list) -> str:
        """
        Assemble les chunks en texte de contexte pour le LLM.

        Prioritise les chunks les plus specifiques au type de question.

        Args:
            chunks: resultats retournes par RetrieverAgent.run()

        Returns:
            Texte de contexte formate
        """
        parties = []
        for i, chunk in enumerate(chunks, 1):
            nom = chunk["metadata"].get("denomination", "Medicament inconnu")
            type_c = chunk["metadata"].get("type_chunk", "general")
            score = chunk["score"]
            parties.append(
                f"[Extrait {i} | {nom} | type={type_c} | pertinence={score:.0%}]\n"
                f"{chunk['contenu']}"
            )
        return "\n\n".join(parties)

    def run(self, question: str, chunks: list, historique: list = None) -> str:
        """
        Execution complete de l'Agent 2.

        Args:
            question: question originale de l'utilisateur
            chunks: resultats de RetrieverAgent (liste de dicts avec scores)
            historique: historique des messages de la conversation (Bonus A)

        Returns:
            Reponse generee par le LLM
        """
        contexte = self._construire_contexte(chunks)

        prompt_utilisateur = (
            "Voici des extraits cliniques de la Base de Donnees Publique des Medicaments (BDPM) :\n\n"
            f"--- CONTEXTE BDPM ---\n{contexte}\n--- FIN CONTEXTE ---\n\n"
            f"Question actuelle : {question}\n\n"
            "Reponds a la question actuelle en te basant UNIQUEMENT sur le contexte fourni ci-dessus et sur l'historique de notre conversation."
        )

        messages = [{"role": "system", "content": self.PROMPT_SYSTEME}]
        
        # Ajout de l'historique recent (Bonus A) - max 3 derniers echanges (6 messages)
        if historique:
            messages.extend(historique[-6:])

        messages.append({"role": "user", "content": prompt_utilisateur})

        reponse = self.client.chat.completions.create(
            model=MODELE_LLM,
            messages=messages,
            temperature=0.2,
            max_tokens=1024,
        )
        return reponse.choices[0].message.content
