"""Pilote par tete : effet d'une injection sur le graphe d'attention,
tete par tete, via ``spectral-trust``.

Regle cardinale, valable dans tout le paquet : **aucune moyenne de lambda_2
entre tetes dans un resultat principal**. Chaque (couche, tete) est une
observation conservee individuellement. Toute agregation entre tetes est
etiquetee SECONDAIRE et accompagnee de la distribution complete.

Repartition declaree des calculs :

* ``spectral-trust`` -- symetrisation, laplacien, spectre, metriques par
  tete (``per_head_metrics``). Jamais reimplemente ici.
* local -- masques de regions sur le graphe COMPLET, coupes, volumes,
  coupe normalisee, flux orientes, appariement, tests, figures.
* ``token_span`` n'est **jamais** utilise : il extrait un sous-graphe induit
  et supprime precisement les aretes entre segments que l'on veut mesurer.
"""
