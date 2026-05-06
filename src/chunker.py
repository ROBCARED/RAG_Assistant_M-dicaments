"""
chunker.py - Etape 2 : Decoupage en chunks specialises

Strategie : 4 types de chunks par medicament, chacun centre sur une dimension :
  - 'general'        : nom, forme, voies, composition, titulaire
  - 'prescription'   : conditions de prescription et de delivrance
  - 'grossesse'      : informations grossesse et allaitement (section 4.6 RCP)
  - 'contre_indications' : contre-indications (section 4.3 RCP)

Cette approche permet au retriever de trouver le chunk le plus pertinent
selon le type de question posee.
"""

TYPES_CHUNKS = ["general", "prescription", "indications", "grossesse", "contre_indications"]


# ─────────────────────────────────────────────────────────────────────────────
# Fonctions de construction de texte par type
# ─────────────────────────────────────────────────────────────────────────────

def _chunk_general(med: dict) -> str | None:
    """Chunk general : informations de base sur le medicament."""
    compositions = ", ".join(med["compositions"]) if med["compositions"] else "Non renseignee"
    return (
        f"Medicament : {med['denomination']}\n"
        f"Forme pharmaceutique : {med['forme']}\n"
        f"Voie(s) d'administration : {med['voies']}\n"
        f"Composition (principes actifs) : {compositions}\n"
        f"Statut AMM : {med['statut_amm']}\n"
        f"Titulaire : {med['titulaire']}"
    ).strip()


def _chunk_prescription(med: dict) -> str | None:
    """
    Chunk conditions de prescription.
    Retourne None si aucune condition renseignee.
    """
    conditions = med.get("conditions_prescription", [])
    if not conditions:
        return None
    conditions_str = " / ".join(conditions)
    return (
        f"Medicament : {med['denomination']}\n"
        f"Conditions de prescription et de delivrance : {conditions_str}\n"
        f"Forme pharmaceutique : {med['forme']}\n"
        f"Voie(s) d'administration : {med['voies']}"
    ).strip()


def _chunk_indications(med: dict) -> str | None:
    """
    Chunk indications therapeutiques (section 4.1 RCP).
    Retourne None si donnee absente.
    """
    ind = med.get("indications")
    if not ind or len(ind.strip()) < 20:
        return None
    return (
        f"Medicament : {med['denomination']}\n"
        f"Indications therapeutiques (symptomes et maladies traites) :\n{ind}"
    ).strip()


def _chunk_grossesse(med: dict) -> str | None:
    """
    Chunk grossesse et allaitement (section 4.6 RCP).
    Retourne None si donnee absente.
    """
    grossesse = med.get("grossesse_allaitement")
    if not grossesse or len(grossesse.strip()) < 20:
        return None
    return (
        f"Medicament : {med['denomination']}\n"
        f"Grossesse et allaitement :\n{grossesse}"
    ).strip()


def _chunk_contre_indications(med: dict) -> str | None:
    """
    Chunk contre-indications (section 4.3 RCP).
    Retourne None si donnee absente.
    """
    ci = med.get("contre_indications")
    if not ci or len(ci.strip()) < 20:
        return None
    return (
        f"Medicament : {med['denomination']}\n"
        f"Contre-indications :\n{ci}"
    ).strip()


# ─────────────────────────────────────────────────────────────────────────────
# Decoupage en sous-chunks si necessaire
# ─────────────────────────────────────────────────────────────────────────────

def decouper_texte(texte: str, taille_max: int = 600, overlap: int = 60) -> list:
    """
    Decoupe un texte en chunks de taille maximale avec overlap.

    Args:
        texte: texte a decouper
        taille_max: nombre maximum de caracteres par chunk
        overlap: chevauchement en caracteres entre chunks adjacents

    Returns:
        Liste de chunks textuels
    """
    if len(texte) <= taille_max:
        return [texte]

    chunks = []
    debut = 0

    while debut < len(texte):
        fin = debut + taille_max
        if fin < len(texte):
            # Cherche une coupure propre (espace ou saut de ligne)
            for char in ["\n", " "]:
                pos = texte.rfind(char, debut, fin)
                if pos > debut:
                    fin = pos
                    break

        chunk = texte[debut:fin].strip()
        if chunk:
            chunks.append(chunk)

        nouveau_debut = fin - overlap
        if nouveau_debut <= debut:
            debut = fin  # Force l'avance pour eviter une boucle infinie
        else:
            debut = nouveau_debut

    return chunks


# ─────────────────────────────────────────────────────────────────────────────
# Fonction principale
# ─────────────────────────────────────────────────────────────────────────────

def chunker_medicaments(medicaments: list, taille_max: int = 600, overlap: int = 60) -> list:
    """
    Convertit la liste de medicaments en chunks specialises avec metadonnees.

    Pour chaque medicament, cree jusqu'a 5 chunks selon les donnees disponibles :
    - 'general'            : toujours present
    - 'prescription'       : si conditions renseignees
    - 'indications'        : si section 4.1 RCP disponible
    - 'grossesse'          : si section 4.6 RCP disponible
    - 'contre_indications' : si section 4.3 RCP disponible

    Args:
        medicaments: liste de dicts medicaments (depuis loader.py)
        taille_max: taille max d'un chunk en caracteres
        overlap: overlap entre sous-chunks adjacents

    Returns:
        Liste de dicts {"contenu": str, "metadata": dict}
    """
    chunks_avec_meta = []

    BUILDERS = {
        "general": _chunk_general,
        "prescription": _chunk_prescription,
        "indications": _chunk_indications,
        "grossesse": _chunk_grossesse,
        "contre_indications": _chunk_contre_indications,
    }

    for med in medicaments:
        for type_chunk, builder in BUILDERS.items():
            texte = builder(med)
            if texte is None:
                continue

            # Decoupe en sous-chunks si le texte est trop long
            sous_chunks = decouper_texte(texte, taille_max=taille_max, overlap=overlap)

            for i, chunk in enumerate(sous_chunks):
                chunks_avec_meta.append({
                    "contenu": chunk,
                    "metadata": {
                        "denomination": med["denomination"],
                        "code_cis": med["code_cis"],
                        "forme": med["forme"],
                        "voies": med["voies"],
                        "compositions": med["compositions"],
                        "conditions_prescription": med.get("conditions_prescription", []),
                        "type_chunk": type_chunk,
                        "chunk_index": i,
                        "total_sous_chunks": len(sous_chunks),
                        "source": "BDPM - Base de Donnees Publique des Medicaments",
                    },
                })

    # Stats par type
    stats = {}
    for c in chunks_avec_meta:
        t = c["metadata"]["type_chunk"]
        stats[t] = stats.get(t, 0) + 1

    print(f"  [OK] {len(chunks_avec_meta)} chunks crees :")
    for t, n in stats.items():
        print(f"       - {t:20s} : {n} chunks")

    return chunks_avec_meta


if __name__ == "__main__":
    # Test rapide
    exemple_med = {
        "code_cis": "12345678",
        "denomination": "DOLIPRANE 1000 mg, comprime",
        "forme": "comprime",
        "voies": "orale",
        "statut_amm": "Autorisation active",
        "etat_commercialisation": "Commercialisee",
        "titulaire": "SANOFI AVENTIS FRANCE",
        "compositions": ["PARACETAMOL 1000 mg"],
        "conditions_prescription": ["Medicament non soumis a prescription medicale"],
        "contre_indications": "Hypersensibilite au paracetamol. Insuffisance hepatocellulaire severe.",
        "grossesse_allaitement": "Le paracetamol peut etre utilise pendant la grossesse a la dose minimale efficace.",
    }

    chunks = chunker_medicaments([exemple_med])
    print(f"\nTotal : {len(chunks)} chunks")
    for c in chunks:
        print(f"\n--- [{c['metadata']['type_chunk']}] ---")
        print(c["contenu"])
