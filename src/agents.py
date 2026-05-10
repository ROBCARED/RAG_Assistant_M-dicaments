import os
import numpy as np
from groq import Groq
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer, CrossEncoder

load_dotenv()

SEUIL_CONFIANCE = 0.35
K_RESULTATS = 5
K_INITIAL = 15
MODELE_LLM = "llama-3.3-70b-versatile"
MODELE_RERANKER = "cross-encoder/mmarco-mMiniLMv2-L12-H384-v1"


def _charger_prompt_systeme() -> str:
    chemin = os.path.join(os.path.dirname(__file__), "..", "context.txt")
    with open(chemin, "r", encoding="utf-8") as f:
        return f.read().strip()


class RetrieverAgent:

    def __init__(self, modele: SentenceTransformer, index, chunks: list, avec_reformulation: bool = True, avec_reranking: bool = True):
        self.modele = modele
        self.index = index
        self.chunks = chunks
        self.avec_reformulation = avec_reformulation
        self.avec_reranking = avec_reranking
        self.client_groq = Groq(api_key=os.getenv("GROQ_API_KEY"))
        self.reranker = CrossEncoder(MODELE_RERANKER) if avec_reranking else None

    def reformuler_question(self, question: str, historique: list = None) -> str:
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
                model=MODELE_LLM,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.1,
                max_tokens=150,
            )
            return reponse.choices[0].message.content.strip()
        except Exception:
            return question

    def rechercher(self, question: str, k: int = K_RESULTATS) -> list:
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
        if not resultats:
            return False
        return resultats[0]["score"] >= SEUIL_CONFIANCE

    def reranker_resultats(self, question: str, resultats: list) -> list:
        if not resultats:
            return resultats
        pairs = [(question, r["contenu"]) for r in resultats]
        scores = self.reranker.predict(pairs)
        for chunk, score in zip(resultats, scores):
            chunk["score_rerank"] = float(score)
        return sorted(resultats, key=lambda x: x["score_rerank"], reverse=True)

    def run(self, question: str, historique: list = None) -> dict:
        question_recherche = question
        reformulee = False

        if self.avec_reformulation:
            question_reformulee = self.reformuler_question(question, historique)
            if question_reformulee and question_reformulee != question:
                question_recherche = question_reformulee
                reformulee = True

        k_fetch = K_INITIAL if self.avec_reranking else K_RESULTATS
        resultats = self.rechercher(question_recherche, k=k_fetch)

        confiance_ok = self.confiance_suffisante(resultats)
        meilleur_score = resultats[0]["score"] if resultats else 0.0

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


class GeneratorAgent:

    PROMPT_SYSTEME = _charger_prompt_systeme()

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
        contexte = self._construire_contexte(chunks)

        sources_list = []
        for c in chunks[:2]:
            nom = c["metadata"].get("denomination")
            sec = self._get_section_label(c["metadata"].get("type_chunk"))
            sources_list.append(f"Notice officielle {nom} — section \"{sec}\"")
        sources_str = " / ".join(sources_list)

        prompt_utilisateur = (
            "Voici les extraits cliniques de la BDPM :\n\n"
            f"--- CONTEXTE ---\n{contexte}\n--- FIN CONTEXTE ---\n\n"
            f"Question : {question}\n\n"
            f"Rappelle-toi : tu dois citer les sources sous ce format : Source : {sources_str}\n"
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
