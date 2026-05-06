import requests
import re
import sys

r = requests.get('https://base-donnees-publique.medicaments.gouv.fr/telechargement', timeout=15)
print("Status:", r.status_code)
# Cherche tous les liens href
links = re.findall(r'href="([^"]+)"', r.text)
for l in links:
    if 'bdpm' in l.lower() or 'CIS' in l or 'fichier' in l:
        print(l)
