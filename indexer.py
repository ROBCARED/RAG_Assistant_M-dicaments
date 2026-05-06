"""
indexer.py - Script d'indexation (a lancer une seule fois)

Pipeline complet : Chargement -> Chunking -> Embedding -> Sauvegarde FAISS
A relancer uniquement si les donnees BDPM changent.

Usage :
    python indexer.py
"""

import sys
import io
import time

# Force UTF-8 output on Windows (evite les erreurs cp1252)
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

from src.loader import charger_medicaments
from src.chunker import chunker_medicaments
from src.embedder import charger_modele, creer_index, sauvegarder_index


def main():
    debut = time.time()
    print("=" * 60)
    print("   [INDEXATION] Assistant Medicaments RAG")
    print("=" * 60)

    # Etape 1 : Chargement des donnees BDPM
    print("\n[1/4] Chargement des donnees BDPM...")
    medicaments = charger_medicaments()
    print(f"  -> {len(medicaments)} medicaments charges")

    # Etape 2 : Chunking
    print("\n[2/4] Decoupage en chunks...")
    chunks = chunker_medicaments(medicaments, taille_max=1000, overlap=100)
    print(f"  -> {len(chunks)} chunks crees")

    # Etape 3 : Chargement du modele d'embedding
    print("\n[3/4] Chargement du modele d'embedding...")
    modele = charger_modele()

    # Etape 4 : Creation et sauvegarde de l'index FAISS
    print("\n[4/4] Creation et sauvegarde de l'index FAISS...")
    index = creer_index(chunks, modele)
    sauvegarder_index(index, chunks)

    # Resume
    duree = time.time() - debut
    print("\n" + "=" * 60)
    print(f"   [OK] Indexation terminee en {duree:.1f}s")
    print(f"   [i]  {len(medicaments)} medicaments -> {len(chunks)} chunks -> index FAISS")
    print("   [>]  Lance maintenant : python rag.py")
    print("=" * 60)


if __name__ == "__main__":
    main()
