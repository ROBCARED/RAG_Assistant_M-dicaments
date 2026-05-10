import os
import csv
import requests
import re
import zipfile
from tqdm import tqdm

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data")

URL_RCP_ZIP = "https://base-donnees-publique.medicaments.gouv.fr/download/file/CIS_RCP.zip"

URLS = {
    "CIS_bdpm.txt": "https://base-donnees-publique.medicaments.gouv.fr/download/file/CIS_bdpm.txt",
    "CIS_COMPO_bdpm.txt": "https://base-donnees-publique.medicaments.gouv.fr/download/file/CIS_COMPO_bdpm.txt",
    "CIS_CPD_bdpm.txt": "https://base-donnees-publique.medicaments.gouv.fr/download/file/CIS_CPD_bdpm.txt",
}

MOTS_CLES = [
    "doliprane", "dafalgan", "efferalgan", "paracetamol", "paracétamol",
    "ibuprofene", "ibuprofène", "advil", "nurofen",
    "aspirine", "aspirin", "aspegic", "aspégic",
    "amoxicilline", "augmentin",
    "smecta", "imodium",
    "ventoline", "becotide",
    "omeprazole", "oméprazole", "inexium",
    "metformine", "glucophage",
]


def telecharger_fichier(nom: str, url: str) -> str:
    os.makedirs(DATA_DIR, exist_ok=True)
    chemin = os.path.join(DATA_DIR, nom)

    headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"}

    if os.path.exists(chemin) and os.path.getsize(chemin) > 1000:
        print(f"  [OK] {nom} deja present ({os.path.getsize(chemin)/1024/1024:.1f} Mo)")
        return chemin

    print(f"  [DL] Telechargement de {nom}...")
    try:
        response = requests.get(url, stream=True, timeout=120, headers=headers)
        response.raise_for_status()
        taille_totale = int(response.headers.get("content-length", 0))

        with open(chemin, "wb") as f, tqdm(total=taille_totale, unit="B", unit_scale=True, desc=nom) as barre:
            for chunk in response.iter_content(chunk_size=32768):
                if chunk:
                    f.write(chunk)
                    barre.update(len(chunk))

        if os.path.getsize(chemin) < 1000:
            raise Exception("Le fichier telecharge est trop petit (erreur serveur ?)")

    except Exception as e:
        if os.path.exists(chemin):
            os.remove(chemin)
        print(f"  [!] Erreur DL {nom}: {e}")
        raise

    return chemin


def extraire_rcp_zip():
    chemin_txt = os.path.join(DATA_DIR, "CIS_RCP.txt")

    if os.path.exists(chemin_txt) and os.path.getsize(chemin_txt) > 1000:
        print(f"  [OK] CIS_RCP.txt deja present ({os.path.getsize(chemin_txt)/1024/1024:.1f} Mo)")
        return chemin_txt

    chemin_zip = telecharger_fichier("CIS_RCP.zip", URL_RCP_ZIP)

    if not os.path.exists(chemin_txt) or os.path.getsize(chemin_txt) == 0:
        print(f"  [ZIP] Extraction de {chemin_zip}...")
        with zipfile.ZipFile(chemin_zip, "r") as z:
            z.extractall(DATA_DIR)
        print(f"  [OK] Fichier extrait : {chemin_txt}")
    return chemin_txt


def lire_cis(chemin: str) -> dict:
    medicaments = {}
    with open(chemin, encoding="utf-8", errors="replace") as f:
        reader = csv.reader(f, delimiter="\t")
        for ligne in reader:
            if len(ligne) < 9:
                continue
            code_cis = ligne[0].strip()
            medicaments[code_cis] = {
                "code_cis": code_cis, "denomination": ligne[1].strip(),
                "forme": ligne[2].strip(), "voies": ligne[3].strip(),
                "statut_amm": ligne[4].strip(), "etat_commercialisation": ligne[6].strip(),
                "titulaire": ligne[8].strip(), "compositions": [],
                "conditions_prescription": [], "indications": None,
                "posologie": None, "contre_indications": None,
                "interactions": None,
                "grossesse_allaitement": None, "effets_indesirables": None,
            }
    return medicaments


def lire_compositions(chemin: str, medicaments: dict) -> dict:
    with open(chemin, encoding="utf-8", errors="replace") as f:
        reader = csv.reader(f, delimiter="\t")
        for ligne in reader:
            if len(ligne) < 6:
                continue
            code_cis = ligne[0].strip()
            if code_cis not in medicaments:
                continue
            substance, dosage, unite = ligne[3].strip(), ligne[4].strip(), ligne[5].strip()
            if (ligne[6] if len(ligne) > 6 else "") == "SA":
                entry = f"{substance} {dosage} {unite}".strip()
                if entry not in medicaments[code_cis]["compositions"]:
                    medicaments[code_cis]["compositions"].append(entry)
    return medicaments


def lire_conditions_prescription(chemin: str, medicaments: dict) -> dict:
    with open(chemin, encoding="utf-8", errors="replace") as f:
        reader = csv.reader(f, delimiter="\t")
        for ligne in reader:
            if len(ligne) < 2:
                continue
            code_cis, cond = ligne[0].strip(), ligne[1].strip()
            if code_cis in medicaments and cond not in medicaments[code_cis]["conditions_prescription"]:
                medicaments[code_cis]["conditions_prescription"].append(cond)
    return medicaments


def lire_rcp_fichier(chemin: str, medicaments: dict) -> dict:
    print("  [i] Lecture des notices completes (Latin-1)...")
    with open(chemin, encoding="latin-1", errors="replace") as f:
        reader = csv.reader(f, delimiter="\t")
        for ligne in reader:
            if len(ligne) < 2:
                continue
            code_cis = ligne[0].strip()
            if code_cis not in medicaments:
                continue
            texte = ligne[1]

            def extraire(num):
                pattern = rf"(?si){re.escape(num)}[\s\.]+(?:[\w\s,']+)[:\.\-\n](.*?)(?=\n\s*\d\.\d|\n\s*\d\.)"
                match = re.search(pattern, texte)
                if match:
                    res = re.sub(r'<[^>]+>', '', match.group(1))
                    res = re.sub(r'\s+', ' ', res).strip()
                    return res[:3000] if len(res) > 10 else None
                return None

            medicaments[code_cis]["indications"] = extraire("4.1")
            medicaments[code_cis]["posologie"] = extraire("4.2")
            medicaments[code_cis]["contre_indications"] = extraire("4.3")
            medicaments[code_cis]["interactions"] = extraire("4.5")
            medicaments[code_cis]["grossesse_allaitement"] = extraire("4.6")
            medicaments[code_cis]["effets_indesirables"] = extraire("4.8")
    return medicaments


API_BDPM_URL = "https://api.medicaments.gouv.fr/1.0/medicaments"


def charger_depuis_api() -> list:
    print("  [API] Interrogation de l'API BDPM...")
    medicaments = []
    seen_cis = set()

    for mot in MOTS_CLES:
        try:
            resp = requests.get(
                API_BDPM_URL,
                params={"denomination": mot},
                timeout=15,
                headers={"User-Agent": "RAG-Assistant-Medicaments/1.0"},
            )
            if resp.status_code != 200:
                continue
            for item in resp.json():
                cis = str(item.get("codeCIS", ""))
                if not cis or cis in seen_cis:
                    continue
                seen_cis.add(cis)
                medicaments.append({
                    "code_cis": cis,
                    "denomination": item.get("denomination", ""),
                    "forme": item.get("formePharmaceutique", ""),
                    "voies": item.get("voiesAdministration", ""),
                    "statut_amm": item.get("statutAMM", ""),
                    "etat_commercialisation": item.get("etatCommercialisation", "Commercialisée"),
                    "titulaire": item.get("titulaire", ""),
                    "compositions": [],
                    "conditions_prescription": [],
                    "indications": None,
                    "posologie": None,
                    "contre_indications": None,
                    "interactions": None,
                    "grossesse_allaitement": None,
                    "effets_indesirables": None,
                })
        except Exception as e:
            print(f"  [!] API: erreur pour '{mot}': {e}")

    print(f"  [API] {len(medicaments)} medicaments recuperes")
    return medicaments


def _enrichir_depuis_scraper(medicaments: dict) -> dict:
    from src.rcp_scraper import scraper_tous_rcp
    cis_codes = list(medicaments.keys())
    cache = scraper_tous_rcp(cis_codes)
    for cis, data in cache.items():
        if cis not in medicaments:
            continue
        for champ in ("indications", "posologie", "contre_indications", "interactions", "grossesse_allaitement", "effets_indesirables"):
            if data.get(champ):
                medicaments[cis][champ] = data[champ]
    return medicaments


def charger_medicaments() -> list:
    print("\n[BDPM] Chargement des donnees...")
    telecharger_fichier("CIS_bdpm.txt", URLS["CIS_bdpm.txt"])
    telecharger_fichier("CIS_COMPO_bdpm.txt", URLS["CIS_COMPO_bdpm.txt"])
    telecharger_fichier("CIS_CPD_bdpm.txt", URLS["CIS_CPD_bdpm.txt"])

    medicaments = lire_cis(os.path.join(DATA_DIR, "CIS_bdpm.txt"))
    medicaments = lire_compositions(os.path.join(DATA_DIR, "CIS_COMPO_bdpm.txt"), medicaments)
    medicaments = lire_conditions_prescription(os.path.join(DATA_DIR, "CIS_CPD_bdpm.txt"), medicaments)

    commercialises = {cis: m for cis, m in medicaments.items()
                      if "Commercialis" in m["etat_commercialisation"]
                      and any(mot in m["denomination"].lower() for mot in MOTS_CLES)}

    try:
        chemin_rcp = extraire_rcp_zip()
        commercialises = lire_rcp_fichier(chemin_rcp, commercialises)
    except Exception as e:
        print(f"  [!] ZIP RCP indisponible ({e})")
        print("  [>] Fallback : scraping des RCP individuels (plus lent)...")
        commercialises = _enrichir_depuis_scraper(commercialises)

    return list(commercialises.values())
