import sys
import io
import os
import argparse
from dotenv import load_dotenv

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

load_dotenv()

if not os.getenv("GROQ_API_KEY"):
    print("ERREUR : GROQ_API_KEY manquante dans le fichier .env")
    sys.exit(1)

from src.embedder import charger_modele, charger_index
from src.agents import RetrieverAgent, GeneratorAgent


def afficher_sources(chunks: list):
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
    print("   ASSISTANT MEDICAMENTS — RAG (BDPM x Groq)")
    print("=" * 65)
    print()

    print("[DEMARRAGE] Chargement de la base de connaissances...")
    try:
        modele = charger_modele()
        index, chunks = charger_index()
    except FileNotFoundError as e:
        print(f"\nERREUR : {e}")
        sys.exit(1)

    agent_retriever = RetrieverAgent(
        modele=modele,
        index=index,
        chunks=chunks,
        avec_reformulation=avec_reformulation,
    )
    agent_generateur = GeneratorAgent()

    mode = "avec reformulation" if avec_reformulation else "standard"
    print(f"\n[OK] Systeme pret — mode : {mode}")
    print("   Commandes : 'sources' pour voir les extraits, 'quit' pour quitter")
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

        print("\n  [Retriever] Recherche en cours...")
        resultat_retriever = agent_retriever.run(question, historique=historique_conversation)

        if resultat_retriever["reformulee"]:
            print(f'  [~] Question reformulee : "{resultat_retriever["question_recherche"]}"')

        score = resultat_retriever["meilleur_score"]
        print(f"  [i] Meilleur score de similarite : {score:.1%}")

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

        types_trouves = list(set(c["metadata"].get("type_chunk", "?") for c in chunks_pertinents))
        print(f"  [i] Sections trouvees : {', '.join(types_trouves)}")
        print(f"  [i] {len(chunks_pertinents)} extraits recuperes")

        print("\n  [Generateur] Generation de la reponse...")
        reponse = agent_generateur.run(question, chunks_pertinents, historique=historique_conversation)

        historique_conversation.append({"role": "user", "content": question})
        historique_conversation.append({"role": "assistant", "content": reponse})

        print("\n" + "-" * 65)
        print(reponse)
        print("-" * 65)
        print("  [i] Tapez 'sources' pour voir les extraits utilises")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Assistant Medicaments RAG")
    parser.add_argument("--no-bonus", action="store_true", help="Desactive la reformulation de question")
    args = parser.parse_args()
    main(avec_reformulation=not args.no_bonus)
