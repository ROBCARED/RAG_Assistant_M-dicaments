TYPES_CHUNKS = ["general", "prescription", "indications", "posologie", "effets_indesirables", "grossesse", "contre_indications", "interactions"]


def _chunk_general(med: dict) -> str | None:
    compositions = ", ".join(med["compositions"]) if med["compositions"] else "Non renseignee"
    return (
        f"Medicament : {med['denomination']}\n"
        f"Forme pharmaceutique : {med['forme']}\n"
        f"Voie(s) d'administration : {med['voies']}\n"
        f"Composition (principes actifs) : {compositions}\n"
        f"Titulaire : {med['titulaire']}"
    ).strip()


def _chunk_prescription(med: dict) -> str | None:
    conditions = med.get("conditions_prescription", [])
    if not conditions:
        return None
    return (
        f"Medicament : {med['denomination']}\n"
        f"Conditions de prescription et de delivrance : {' / '.join(conditions)}"
    ).strip()


def _chunk_indications(med: dict) -> str | None:
    ind = med.get("indications")
    if not ind or len(ind.strip()) < 20:
        return None
    return f"Medicament : {med['denomination']}\nSection : Indications therapeutiques\nContenu : {ind}"


def _chunk_posologie(med: dict) -> str | None:
    poso = med.get("posologie")
    if not poso or len(poso.strip()) < 20:
        return None
    return f"Medicament : {med['denomination']}\nSection : Posologie et mode d'administration\nContenu : {poso}"


def _chunk_effets_indesirables(med: dict) -> str | None:
    effets = med.get("effets_indesirables")
    if not effets or len(effets.strip()) < 20:
        return None
    return f"Medicament : {med['denomination']}\nSection : Effets indesirables\nContenu : {effets}"


def _chunk_grossesse(med: dict) -> str | None:
    grossesse = med.get("grossesse_allaitement")
    if not grossesse or len(grossesse.strip()) < 20:
        return None
    return f"Medicament : {med['denomination']}\nSection : Fertilite, grossesse et allaitement\nContenu : {grossesse}"


def _chunk_contre_indications(med: dict) -> str | None:
    ci = med.get("contre_indications")
    if not ci or len(ci.strip()) < 20:
        return None
    return f"Medicament : {med['denomination']}\nSection : Contre-indications\nContenu : {ci}"


def _chunk_interactions(med: dict) -> str | None:
    inter = med.get("interactions")
    if not inter or len(inter.strip()) < 20:
        return None
    return f"Medicament : {med['denomination']}\nSection : Interactions avec d'autres medicaments\nContenu : {inter}"


def decouper_texte(texte: str, taille_max: int = 600, overlap: int = 60) -> list:
    if len(texte) <= taille_max:
        return [texte]
    chunks = []
    debut = 0
    while debut < len(texte):
        fin = debut + taille_max
        if fin < len(texte):
            for char in ["\n", " "]:
                pos = texte.rfind(char, debut, fin)
                if pos > debut:
                    fin = pos
                    break
        chunks.append(texte[debut:fin].strip())
        debut = fin - overlap
        if debut >= fin:
            debut = fin
    return chunks


def chunker_medicaments(medicaments: list, taille_max: int = 600, overlap: int = 60) -> list:
    chunks_avec_meta = []
    BUILDERS = {
        "general": _chunk_general,
        "prescription": _chunk_prescription,
        "indications": _chunk_indications,
        "posologie": _chunk_posologie,
        "effets_indesirables": _chunk_effets_indesirables,
        "grossesse": _chunk_grossesse,
        "contre_indications": _chunk_contre_indications,
        "interactions": _chunk_interactions,
    }
    for med in medicaments:
        for type_chunk, builder in BUILDERS.items():
            texte = builder(med)
            if not texte:
                continue
            sous_chunks = decouper_texte(texte, taille_max=taille_max, overlap=overlap)
            for chunk in sous_chunks:
                chunks_avec_meta.append({
                    "contenu": chunk,
                    "metadata": {
                        "denomination": med["denomination"],
                        "code_cis": med["code_cis"],
                        "type_chunk": type_chunk,
                        "source": f"Notice officielle {med['denomination']}",
                    },
                })
    return chunks_avec_meta
