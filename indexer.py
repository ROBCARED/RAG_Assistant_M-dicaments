import sys
import io
import time

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

from src.loader import charger_medicaments
from src.chunker import chunker_medicaments
from src.embedder import charger_modele, creer_index, sauvegarder_index


def main():
    debut = time.time()
    print("=" * 60)
    print("   Assistant Medicaments — Indexation")
    print("=" * 60)

    print("\n[1/4] Chargement des donnees BDPM...")
    medicaments = charger_medicaments()
    print(f"  -> {len(medicaments)} medicaments charges")

    print("\n[2/4] Decoupage en chunks...")
    chunks = chunker_medicaments(medicaments, taille_max=1000, overlap=100)
    print(f"  -> {len(chunks)} chunks crees")

    print("\n[3/4] Chargement du modele d'embedding...")
    modele = charger_modele()

    print("\n[4/4] Creation et sauvegarde de l'index FAISS...")
    index = creer_index(chunks, modele)
    sauvegarder_index(index, chunks)

    duree = time.time() - debut
    print("\n" + "=" * 60)
    print(f"   Indexation terminee en {duree:.1f}s")
    print(f"   {len(medicaments)} medicaments -> {len(chunks)} chunks -> index FAISS")
    print("   Lance maintenant : python rag.py")
    print("=" * 60)


if __name__ == "__main__":
    main()
