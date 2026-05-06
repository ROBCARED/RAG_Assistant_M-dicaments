"""
rag.py - Interface en ligne de commande (CLI)

Orchestrateur du systeme RAG a deux agents :
  - Agent 1 (RetrieverAgent) : recherche + reformulation + confiance
  - Agent 2 (GeneratorAgent) : generation via Groq

Usage :
    python rag.py              # Mode avec reformulation (Bonus C actif)
    python rag.py --no-bonus   # Sans reformulation de question
"""

import sys
import io
import os
import argparse
from dotenv import load_dotenv

# Force UTF-8 output on Windows
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

load_dotenv()

if not os.getenv("GROQ_API_KEY"):
    print("ERREUR : GROQ_API_KEY manquante dans le fichier .env")
    sys.exit(1)

from src.embedder import charger_modele, charger_index
from src.agents import RetrieverAgent, GeneratorAgent


def afficher_sources(chunks: list):
    """Affiche les details des chunks utilises avec leur score."""
    print("\n[SOURCES CONSULTEES]")
    for i, chunk in enumerate(chunks, 1):
        nom = chunk["metadata"].get("denomination", "Inconnu")
        type_c = chunk["metadata"].get("type_chunk", "?")
        score = chunk["score"]
        barre = "#" * int(score * 20) + "." * (20 - int(score * 20))
        print(f"  {i}. [{type_c:20s}] {nom}")
        print(f"     Pertinence : [{barre}] {score:.1%}")


def main(avec_reformulation: bool = True):
    print("=" * 65)
    print("   ASSISTANT MEDICAMENTS - RAG 2 AGENTS (BDPM x Groq)")
    print("=" * 65)
    print()
    print("   Architecture :")
    print("   [Utilisateur] -> [Agent 1 : Retriever] -> [Agent 2 : Generateur] -> [Reponse]")
    print()

    # ── Chargement des composants ─────────────────────────────────
    print("[DEMARRAGE] Chargement de la base de connaissances...")
    try:
        modele = charger_modele()
        index, chunks = charger_index()
    except FileNotFoundError as e:
        print(f"\nERREUR : {e}")
        sys.exit(1)

    # ── Initialisation des agents ─────────────────────────────────
    agent_retriever = RetrieverAgent(
        modele=modele,
        index=index,
        chunks=chunks,
        avec_reformulation=avec_reformulation,
    )
    agent_generateur = GeneratorAgent()

    mode = "avec reformulation (Bonus C)" if avec_reformulation else "standard"
    print(f"\n[OK] Systeme RAG 2-agents pret - Mode : {mode}")
    print("   Commandes speciales : 'sources', 'quit'")
    print("-" * 65)

    derniers_chunks = []
    historique_conversation = []

    while True:
        try:
            question = input("\n[?] Votre question : ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\n\nAu revoir !")
            break

        if not question:
            continue

        if question.lower() in ("quit", "exit", "q"):
            print("\nAu revoir !")
            break

        if question.lower() == "sources" and derniers_chunks:
            afficher_sources(derniers_chunks)
            continue

        # ── Agent 1 : Retrieval ───────────────────────────────────
        print("\n  [Agent 1 - Retriever] Recherche en cours...")

        resultat_retriever = agent_retriever.run(question, historique=historique_conversation)

        if resultat_retriever["reformulee"]:
            print(f'  [~] Question reformulee : "{resultat_retriever["question_recherche"]}"')

        score = resultat_retriever["meilleur_score"]
        print(f"  [i] Meilleur score de similarite : {score:.1%}")

        # Bonus B : score de confiance insuffisant
        if not resultat_retriever["confiance_ok"]:
            print("\n  [!] Score de confiance trop faible.")
            print("  Cette question depasse le perimetre des medicaments indexes dans ma base de donnees.")
            print("  Medicaments couverts : Doliprane, Dafalgan, Efferalgan, Advil, Nurofen, Aspirin,")
            print("  Aspegic, Amoxicilline, Augmentin, Smecta, Imodium, Ventoline, Becotide,")
            print("  Omeprazole, Inexium, Metformine, Glucophage.")
            print("\n  Ces informations ne remplacent pas l'avis d'un professionnel de sante.")
            continue

        chunks_pertinents = resultat_retriever["resultats"]
        derniers_chunks = chunks_pertinents

        # Affiche les types de chunks trouves
        types_trouves = list(set(c["metadata"].get("type_chunk", "?") for c in chunks_pertinents))
        print(f"  [i] Types de chunks : {', '.join(types_trouves)}")
        print(f"  [i] {len(chunks_pertinents)} chunks recuperes")

        # ── Agent 2 : Generation ──────────────────────────────────
        print("\n  [Agent 2 - Generateur] Generation de la reponse (Groq)...")
        reponse = agent_generateur.run(question, chunks_pertinents, historique=historique_conversation)

        # Ajout a l'historique (Bonus A)
        historique_conversation.append({"role": "user", "content": question})
        historique_conversation.append({"role": "assistant", "content": reponse})

        print("\n" + "-" * 65)
        print(reponse)
        print("-" * 65)
        print("  [i] Tapez 'sources' pour voir les extraits utilises")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Assistant Medicaments RAG 2-agents")
    parser.add_argument(
        "--no-bonus",
        action="store_true",
        help="Desactive la reformulation de question (Bonus C)",
    )
    args = parser.parse_args()
    main(avec_reformulation=not args.no_bonus)
