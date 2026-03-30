"""
Récupération des données de marché énergétique européen.

Ce module est responsable de l'acquisition des données brutes depuis
les sources externes : APIs de marchés (ENTSO-E, EEX, EPEX), flux RSS,
fichiers CSV ou bases de données. Il expose des fonctions permettant
de télécharger et stocker localement les prix spot, futures, volumes
et données fondamentales (production, consommation, météo).
"""
