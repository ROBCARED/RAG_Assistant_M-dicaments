"""
loader.py - Etape 1 : Chargement des donnees BDPM

Telecharge et parse les fichiers CSV de la BDPM :
- CIS_bdpm.txt        : specialites (nom, forme, voies, statut)
- CIS_COMPO_bdpm.txt  : compositions (principes actifs, dosages)
- CIS_CPD_bdpm.txt    : conditions de prescription et de delivrance

Enrichit ensuite chaque medicament avec les donnees RCP scraped :
- Contre-indications (section 4.3 du RCP)
- Grossesse et allaitement (section 4.6 du RCP)
"""

import os
import csv
import requests
from tqdm import tqdm

# Repertoire de stockage des donnees
DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data")

# URLs des fichiers BDPM
BASE_URL = "https://base-donnees-publique.medicaments.gouv.fr"

URLS = {
    "CIS_bdpm.txt": f"{BASE_URL}/download/file/CIS_bdpm.txt",
    "CIS_COMPO_bdpm.txt": f"{BASE_URL}/download/file/CIS_COMPO_bdpm.txt",
    "CIS_CPD_bdpm.txt": f"{BASE_URL}/download/file/CIS_CPD_bdpm.txt",
}


def telecharger_fichier(nom: str, url: str) -> str:
    """
    Telecharge un fichier BDPM si non present localement.

    Args:
        nom: nom du fichier a sauvegarder
        url: URL de telechargement

    Returns:
        Chemin local vers le fichier telecharge
    """
    os.makedirs(DATA_DIR, exist_ok=True)
    chemin = os.path.join(DATA_DIR, nom)

    if os.path.exists(chemin):
        print(f"  [OK] {nom} deja present (skip telechargement)")
        return chemin

    print(f"  [DL] Telechargement de {nom}...")
    response = requests.get(url, stream=True, timeout=60)
    response.raise_for_status()

    taille_totale = int(response.headers.get("content-length", 0))
    with open(chemin, "wb") as f, tqdm(
        total=taille_totale, unit="B", unit_scale=True, desc=nom
    ) as barre:
        for chunk in response.iter_content(chunk_size=8192):
            f.write(chunk)
            barre.update(len(chunk))

    print(f"  [OK] {nom} telecharge")
    return chemin


def lire_cis(chemin: str) -> dict:
    """
    Lit CIS_bdpm.txt : specialites pharmaceutiques.

    Colonnes : CodeCIS, Denomination, Forme, Voies, Statut AMM,
               Type procedure, Etat commercialisation, Date AMM,
               Titulaire, Surveillance
    """
    medicaments = {}
    with open(chemin, encoding="utf-8", errors="replace") as f:
        reader = csv.reader(f, delimiter="\t")
        for ligne in reader:
            if len(ligne) < 9:
                continue
            code_cis = ligne[0].strip()
            medicaments[code_cis] = {
                "code_cis": code_cis,
                "denomination": ligne[1].strip(),
                "forme": ligne[2].strip(),
                "voies": ligne[3].strip(),
                "statut_amm": ligne[4].strip(),
                "etat_commercialisation": ligne[6].strip(),
                "titulaire": ligne[8].strip(),
                # Champs enrichis (remplis par les autres parsers)
                "compositions": [],
                "conditions_prescription": [],
                "indications": None,
                "contre_indications": None,
                "grossesse_allaitement": None,
            }
    return medicaments


def lire_compositions(chemin: str, medicaments: dict) -> dict:
    """
    Lit CIS_COMPO_bdpm.txt : principes actifs et dosages.

    Colonnes : CodeCIS, Element pharm., Code substance,
               Denomination substance, Dosage, Ref dosage,
               Nature composant, Liaison SA/FT

    Filtre sur nature = 'SA' (substance active uniquement).
    """
    with open(chemin, encoding="utf-8", errors="replace") as f:
        reader = csv.reader(f, delimiter="\t")
        for ligne in reader:
            if len(ligne) < 6:
                continue
            code_cis = ligne[0].strip()
            if code_cis not in medicaments:
                continue
            substance = ligne[3].strip()
            dosage = ligne[4].strip()
            unite = ligne[5].strip()
            nature = ligne[6].strip() if len(ligne) > 6 else ""
            if nature == "SA" and substance:
                entry = f"{substance} {dosage} {unite}".strip()
                if entry not in medicaments[code_cis]["compositions"]:
                    medicaments[code_cis]["compositions"].append(entry)
    return medicaments


def lire_conditions_prescription(chemin: str, medicaments: dict) -> dict:
    """
    Lit CIS_CPD_bdpm.txt : conditions de prescription et de delivrance.

    Colonnes : CodeCIS, Condition de prescription ou delivrance

    Exemples de valeurs :
    - 'Liste I'
    - 'Liste II'
    - 'Stupefiant'
    - 'Prescription hospitaliere'
    - 'Prescription reservee aux specialistes en...'
    """
    with open(chemin, encoding="utf-8", errors="replace") as f:
        reader = csv.reader(f, delimiter="\t")
        for ligne in reader:
            if len(ligne) < 2:
                continue
            code_cis = ligne[0].strip()
            if code_cis not in medicaments:
                continue
            condition = ligne[1].strip()
            if condition and condition not in medicaments[code_cis]["conditions_prescription"]:
                medicaments[code_cis]["conditions_prescription"].append(condition)
    return medicaments


def enrichir_avec_rcp(medicaments: dict, cache_rcp: dict) -> dict:
    """
    Enrichit chaque medicament avec les donnees RCP (indications, contre-indications, grossesse).

    Args:
        medicaments: dict {cis -> infos medicament}
        cache_rcp: dict {cis -> {"indications": str, "contre_indications": str, "grossesse_allaitement": str}}

    Returns:
        Dict medicaments enrichi
    """
    n_enrichis = 0
    for cis, med in medicaments.items():
        if cis in cache_rcp:
            rcp = cache_rcp[cis]
            med["indications"] = rcp.get("indications")
            med["contre_indications"] = rcp.get("contre_indications")
            med["grossesse_allaitement"] = rcp.get("grossesse_allaitement")
            if rcp.get("contre_indications") or rcp.get("grossesse_allaitement") or rcp.get("indications"):
                n_enrichis += 1
    print(f"  [OK] {n_enrichis} medicaments enrichis avec donnees RCP")
    return medicaments


def charger_medicaments(avec_rcp: bool = True) -> list:
    """
    Point d'entree principal.

    Telecharge les fichiers CSV BDPM, les parse, puis enrichit
    optionnellement avec les donnees RCP (contre-indications, grossesse).

    Args:
        avec_rcp: si True, scrape les RCP (lent, ~10min pour tous les medics)

    Returns:
        Liste de dicts medicament avec cles :
        code_cis, denomination, forme, voies, statut_amm,
        etat_commercialisation, titulaire, compositions,
        conditions_prescription, contre_indications, grossesse_allaitement
    """
    print("\n[BDPM] Chargement des donnees...")

    # Telechargement des fichiers CSV
    chemin_cis = telecharger_fichier("CIS_bdpm.txt", URLS["CIS_bdpm.txt"])
    chemin_compo = telecharger_fichier("CIS_COMPO_bdpm.txt", URLS["CIS_COMPO_bdpm.txt"])
    chemin_cpd = telecharger_fichier("CIS_CPD_bdpm.txt", URLS["CIS_CPD_bdpm.txt"])

    # Parsing des specialites
    print("  [i] Lecture des specialites...")
    medicaments = lire_cis(chemin_cis)
    print(f"  [OK] {len(medicaments)} specialites chargees")

    # Ajout des compositions
    print("  [i] Lecture des compositions...")
    medicaments = lire_compositions(chemin_compo, medicaments)

    # Ajout des conditions de prescription
    print("  [i] Lecture des conditions de prescription...")
    medicaments = lire_conditions_prescription(chemin_cpd, medicaments)

    # Filtrage : uniquement les medicaments commercialises et ciblant le corpus restreint du TP
    mots_cles_tp = [
        "doliprane", "dafalgan", "efferalgan", "paracetamol", "paracétamol",
        "ibuprofene", "ibuprofène", "advil", "nurofen",
        "aspirine", "aspirin", "aspegic", "aspégic",
        "amoxicilline", "augmentin",
        "smecta", "imodium",
        "ventoline", "becotide",
        "omeprazole", "oméprazole", "inexium",
        "metformine", "glucophage"
    ]
    
    commercialises = {}
    for cis, m in medicaments.items():
        if "Commercialis" in m["etat_commercialisation"]:
            denom = m["denomination"].lower()
            if any(mot in denom for mot in mots_cles_tp):
                commercialises[cis] = m
                
    print(f"  [OK] {len(commercialises)} medicaments du corpus restreint (TP) retenus")

    # Enrichissement RCP (scraping)
    if avec_rcp:
        from src.rcp_scraper import scraper_tous_rcp
        cis_codes = list(commercialises.keys())
        print(f"\n[RCP] Enrichissement avec donnees RCP ({len(cis_codes)} medicaments)...")
        cache_rcp = scraper_tous_rcp(cis_codes)
        commercialises = enrichir_avec_rcp(commercialises, cache_rcp)

    return list(commercialises.values())


if __name__ == "__main__":
    # Test sans RCP (rapide)
    medicaments = charger_medicaments(avec_rcp=False)
    exemple = medicaments[0]
    print(f"\nExemple : {exemple['denomination']}")
    print(f"  Forme        : {exemple['forme']}")
    print(f"  Voies        : {exemple['voies']}")
    print(f"  Compo        : {', '.join(exemple['compositions'][:3])}")
    print(f"  Prescription : {', '.join(exemple['conditions_prescription'])}")
