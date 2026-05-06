"""
rcp_scraper.py - Scraping des RCP (Resume des Caracteristiques du Produit)

Telecharge et parse les pages RCP de la BDPM pour extraire :
- Section 4.3 : Contre-indications
- Section 4.6 : Grossesse et allaitement

Utilise un cache JSON pour eviter les re-telechargements.
Le scraping est parallelise (10 threads) pour rester rapide.
"""

import os
import json
import requests
import time
from bs4 import BeautifulSoup
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data")
CACHE_FILE = os.path.join(DATA_DIR, "rcp_cache.json")
BASE_URL = "https://base-donnees-publique.medicaments.gouv.fr/affichageDoc.php?specid={cis}&typedoc=R"

MAX_WORKERS = 8
TIMEOUT = 10


import re

def _extraire_section_regex(texte: str, numero_section: str, max_chars: int = 1000) -> str:
    """
    Extrait une section du RCP en utilisant des expressions régulières.
    Plus robuste que la recherche de ligne simple.
    """
    # Pattern : Cherche un début de ligne ou un saut de ligne suivi du numéro de section (ex: 4.1)
    # S'arrête à la section suivante (ex: 4.2) ou à une autre section majeure.
    
    if numero_section == "4.1":
        pattern = r"(?s)4\.1\s*[\.\-]?\s*Indications therapeutiques(.*?)(?=4\.2|4\.3|4\.4|4\.5|4\.6|5\.)"
    elif numero_section == "4.2":
        pattern = r"(?s)4\.2\s*[\.\-]?\s*Posologie\s*(?:et\s*mode\s*d.administration)?(.*?)(?=4\.3|4\.4|4\.5|5\.)"
    elif numero_section == "4.3":
        pattern = r"(?s)4\.3\s*[\.\-]?\s*Contre-indications(.*?)(?=4\.4|4\.5|4\.6|4\.7|4\.8|5\.)"
    elif numero_section == "4.5":
        pattern = r"(?s)4\.5\s*[\.\-]?\s*Interactions\s*(?:avec\s*d.autres\s*m.dicaments)?(.*?)(?=4\.6|4\.7|4\.8|5\.)"
    elif numero_section == "4.6":
        pattern = r"(?s)4\.6\s*[\.\-]?\s*(?:Fertilite, )?grossesse et allaitement(.*?)(?=4\.7|4\.8|4\.9|5\.)"
    elif numero_section == "4.8":
        pattern = r"(?s)4\.8\s*[\.\-]?\s*Effets\s*ind.sirables(.*?)(?=4\.9|5\.)"
    else:
        return None

    match = re.search(pattern, texte, re.IGNORECASE)
    if match:
        contenu = match.group(1).strip()
        # Nettoyage minimal des sauts de ligne multiples
        contenu = re.sub(r'\n+', ' ', contenu)
        contenu = re.sub(r'\s+', ' ', contenu)
        return contenu[:max_chars]
    
    return None

def scraper_rcp_medicament(cis: str) -> dict:
    """
    Scrape le RCP d'un medicament identifie par son code CIS.
    Utilise désormais des REGEX pour une extraction précise.
    """
    url = BASE_URL.format(cis=cis)
    result = {
        "indications": None,
        "posologie": None,
        "contre_indications": None,
        "interactions": None,
        "grossesse_allaitement": None,
        "effets_indesirables": None,
    }

    try:
        r = requests.get(url, timeout=TIMEOUT)
        if r.status_code != 200:
            return result

        soup = BeautifulSoup(r.content, "html.parser")
        texte_complet = soup.get_text(separator="\n")

        result["indications"]           = _extraire_section_regex(texte_complet, "4.1", max_chars=1200)
        result["posologie"]             = _extraire_section_regex(texte_complet, "4.2", max_chars=1500)
        result["contre_indications"]    = _extraire_section_regex(texte_complet, "4.3", max_chars=1200)
        result["interactions"]          = _extraire_section_regex(texte_complet, "4.5", max_chars=1500)
        result["grossesse_allaitement"] = _extraire_section_regex(texte_complet, "4.6", max_chars=1200)
        result["effets_indesirables"]   = _extraire_section_regex(texte_complet, "4.8", max_chars=1500)

    except Exception as e:
        print(f"Erreur scraping CIS {cis}: {e}")

    return result


def charger_cache() -> dict:
    """Charge le cache RCP depuis le disque."""
    if os.path.exists(CACHE_FILE):
        with open(CACHE_FILE, encoding="utf-8") as f:
            return json.load(f)
    return {}


def sauvegarder_cache(cache: dict):
    """Sauvegarde le cache RCP sur le disque."""
    os.makedirs(DATA_DIR, exist_ok=True)
    with open(CACHE_FILE, "w", encoding="utf-8") as f:
        json.dump(cache, f, ensure_ascii=False, indent=2)


def scraper_tous_rcp(cis_codes: list, force: bool = False) -> dict:
    """
    Scrape les RCP de tous les medicaments de la liste.
    Utilise un cache pour ne pas re-telecharger les pages deja scraped.

    Args:
        cis_codes: liste des codes CIS a scraper
        force: si True, ignore le cache et re-scrape tout

    Returns:
        Dict {cis_code -> {"contre_indications": str|None, "grossesse_allaitement": str|None}}
    """
    cache = {} if force else charger_cache()

    # Codes non encore dans le cache
    a_scraper = [cis for cis in cis_codes if cis not in cache]

    if not a_scraper:
        print(f"  [OK] Cache RCP complet ({len(cache)} entrees)")
        return cache

    print(f"  [RCP] Scraping de {len(a_scraper)} pages RCP ({MAX_WORKERS} threads)...")

    def scraper_et_stocker(cis):
        data = scraper_rcp_medicament(cis)
        return cis, data

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futurs = {executor.submit(scraper_et_stocker, cis): cis for cis in a_scraper}
        for futur in tqdm(as_completed(futurs), total=len(a_scraper), desc="RCP"):
            try:
                cis, data = futur.result()
                cache[cis] = data
            except Exception:
                cis = futurs[futur]
                cache[cis] = {"indications": None, "contre_indications": None, "grossesse_allaitement": None}

    sauvegarder_cache(cache)
    trouvees = sum(
        1 for v in cache.values()
        if v.get("contre_indications") or v.get("grossesse_allaitement") or v.get("indications")
    )
    print(f"  [OK] Cache RCP sauvegarde : {trouvees}/{len(cache)} avec donnees")
    return cache


if __name__ == "__main__":
    # Test sur 5 medicaments
    test_cis = ["61266250", "62869109", "69103878"]
    resultats = scraper_tous_rcp(test_cis)
    for cis, data in resultats.items():
        print(f"\n=== CIS {cis} ===")
        print(f"Contre-indications : {(data['contre_indications'] or 'N/A')[:200]}")
        print(f"Grossesse          : {(data['grossesse_allaitement'] or 'N/A')[:200]}")
