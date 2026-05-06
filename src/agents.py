"""
agents.py - Architecture multi-agents (Version Stable)

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

Note: Cette version utilise uniquement l'indexation statique pre-construite.
"""

import os
import numpy as np
from groq import Groq
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer, CrossEncoder

load_dotenv()

# ─────────────────────────────────────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────────────────────────────────────

SEUIL_CONFIANCE = 0.35      # Score minimum pour considerer un resultat pertinent
K_RESULTATS = 5             # Nombre de chunks retournes apres reranking
K_INITIAL = 15              # Nombre de chunks recuperes avant reranking
MODELE_LLM = "llama-3.3-70b-versatile"
MODELE_LLM_RAPIDE = "llama-3.3-70b-versatile"
MODELE_RERANKER = "cross-encoder/mmarco-mMiniLMv2-L12-H384-v1"


# ─────────────────────────────────────────────────────────────────────────────
# Agent 1 : RetrieverAgent
# ─────────────────────────────────────────────────────────────────────────────

class RetrieverAgent:
    """
    Agent 1 : Recherche et recuperation des chunks pertinents.
    """

    def __init__(self, modele: SentenceTransformer, index, chunks: list, avec_reformulation: bool = True, avec_reranking: bool = True):
        """
        Args:
            modele: modele sentence-transformers (meme que celui utilise a l'indexation)
            index: index FAISS charge
            chunks: liste des chunks dans le meme ordre que les vecteurs FAISS
            avec_reformulation: active le Bonus C (reformulation de question)
            avec_reranking: active le reranking Cross-Encoder
        """
        self.modele = modele
        self.index = index
        self.chunks = chunks
        self.avec_reformulation = avec_reformulation
        self.avec_reranking = avec_reranking
        self.client_groq = Groq(api_key=os.getenv("GROQ_API_KEY"))
        self.reranker = CrossEncoder(MODELE_RERANKER) if avec_reranking else None

    def reformuler_question(self, question: str, historique: list = None) -> str:
        """
        Bonus C - Reformule la question en termes pharmaceutiques precis.
        """
        contexte_historique = ""
        if historique:
            contexte_historique = "Historique recent de la conversation :\n"
            for msg in historique[-4:]:
                role = "Utilisateur" if msg["role"] == "user" else "Assistant"
                contenu = msg['content'][:300] + "..." if len(msg['content']) > 300 else msg['content']
                contexte_historique += f"- {role}: {contenu}\n"
            contexte_historique += "\n"

        prompt = (
            "Tu es un expert en pharmacologie. L'utilisateur va te decrire des symptomes ou poser une question.\n"
            f"{contexte_historique}"
            "En tenant compte de l'historique si necessaire, reformule la NOUVELLE demande en termes pharmaceutiques "
            "et medicaux precis pour optimiser une recherche vectorielle.\n"
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
            return question

    def rechercher(self, question: str, k: int = K_RESULTATS) -> list:
        """
        Encode la question et interroge l'index FAISS.
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
        """
        if not resultats:
            return False
        return resultats[0]["score"] >= SEUIL_CONFIANCE

    def reranker_resultats(self, question: str, resultats: list) -> list:
        """
        Reranke les resultats avec un Cross-Encoder et trie par score decroissant.
        Ajoute le champ 'score_rerank' a chaque chunk.
        """
        if not resultats:
            return resultats
        pairs = [(question, r["contenu"]) for r in resultats]
        scores = self.reranker.predict(pairs)
        for chunk, score in zip(resultats, scores):
            chunk["score_rerank"] = float(score)
        return sorted(resultats, key=lambda x: x["score_rerank"], reverse=True)

    def run(self, question: str, historique: list = None) -> dict:
        """
        Execution complete de l'Agent 1.
        """
        question_recherche = question
        reformulee = False

        if self.avec_reformulation:
            question_reformulee = self.reformuler_question(question, historique)
            if question_reformulee and question_reformulee != question:
                question_recherche = question_reformulee
                reformulee = True

        # On recupere plus de candidats si le reranking est actif
        k_fetch = K_INITIAL if self.avec_reranking else K_RESULTATS
        resultats = self.rechercher(question_recherche, k=k_fetch)

        # Le score de confiance est verifie sur les scores FAISS (avant reranking)
        confiance_ok = self.confiance_suffisante(resultats)
        meilleur_score = resultats[0]["score"] if resultats else 0.0

        # Reranking Cross-Encoder puis troncature au top-K
        if self.avec_reranking and self.reranker:
            resultats = self.reranker_resultats(question_recherche, resultats)
        resultats = resultats[:K_RESULTATS]

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
    """

    PROMPT_SYSTEME = """Tu es un Systeme d'Aide a la Decision Medicale (SADM) expert base sur la Base de Donnees Publique des Medicaments (BDPM).

Ton role est de repondre aux questions cliniques en te basant UNIQUEMENT sur le contexte fourni.

CONSIGNES DE REPONSE :
1. Chaque reponse doit OBLIGATOIREMENT inclure cette mention exacte a la fin : "Ces informations ne remplacent pas l'avis d'un professionnel de sante. En cas de doute, consultez votre medecin ou votre pharmacien."
2. Indique CLAIREMENT de quel medicament provient chaque information (ex: "D'apres la notice officielle de l'ibuprofene...").
3. Cite la section precise du RCP (ex: section "Indications", section "Effets indesirables", section "Posologie").
4. Si l'information n'est pas dans le contexte, dis-le EXPLICITEMENT plutot que d'inventer une reponse. Formule : "Cette question depasse le perimetre des medicaments indexes dans ma base de donnees."
5. Si la question concerne PLUSIEURS medicaments (ex: interactions), synthetise les informations de chaque medicament et presente les resultats de maniere croisee.
6. Sois informatif mais systematiquement prudent sur les recommandations medicales.
7. Structure ta reponse avec des tirets (-) pour les listes.

FORMAT DE SORTIE ATTENDU :
D'apres la notice officielle de [Nom Medicament], [resume de la reponse] :
 - [Point 1]
 - [Point 2]
 - [Point 3]

Ces informations ne remplacent pas l'avis d'un professionnel de sante.
En cas de doute, consultez votre medecin ou votre pharmacien.

Source : Notice officielle [Nom Medicament] — section "[Nom Section]" """

    def __init__(self):
        self.client = Groq(api_key=os.getenv("GROQ_API_KEY"))

    def _get_section_label(self, type_chunk: str) -> str:
        mapping = {
            "indications": "Indications therapeutiques",
            "posologie": "Posologie et mode d'administration",
            "contre_indications": "Contre-indications",
            "interactions": "Interactions avec d'autres medicaments",
            "effets_indesirables": "Effets indesirables",
            "grossesse": "Fertilite, grossesse et allaitement",
            "prescription": "Conditions de prescription et de delivrance",
            "general": "Informations generales"
        }
        return mapping.get(type_chunk, "Notice officielle")

    def _construire_contexte(self, chunks: list) -> str:
        """
        Assemble les chunks recuperes en un texte de contexte structure
        pour le LLM, incluant le nom du medicament et la section du RCP.
        """
        parties = []
        for i, chunk in enumerate(chunks, 1):
            nom = chunk["metadata"].get("denomination", "Medicament inconnu")
            type_c = self._get_section_label(chunk["metadata"].get("type_chunk"))
            score = chunk["score"]
            parties.append(
                f"[Extrait {i} — {nom} | Section: {type_c} | Pertinence: {score:.2f}]\n"
                f"{chunk['contenu']}"
            )
        return "\n\n".join(parties)

    def run(self, question: str, chunks: list, historique: list = None) -> str:
        """Execution complete de l'Agent 2."""
        contexte = self._construire_contexte(chunks)
        
        # Generation de la ligne de source pour aider le LLM a suivre le format
        sources_list = []
        for c in chunks[:2]: # On cite les 2 sources principales
            nom = c["metadata"].get("denomination")
            sec = self._get_section_label(c["metadata"].get("type_chunk"))
            sources_list.append(f"Notice officielle {nom} — section \"{sec}\"")
        sources_str = " / ".join(sources_list)

        prompt_utilisateur = (
            "Voici les extraits cliniques de la BDPM :\n\n"
            f"--- CONTEXTE ---\n{contexte}\n--- FIN CONTEXTE ---\n\n"
            f"Question : {question}\n\n"
            f"REPPELLE-TOI : Tu dois citer les sources sous ce format : Source : {sources_str}\n"
            "Reponds maintenant."
        )

        messages = [{"role": "system", "content": self.PROMPT_SYSTEME}]
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
