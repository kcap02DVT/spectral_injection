"""Campagne Open-Prompt-Injection — protocole Spectral-Trust.

Sous-paquet autonome. Il ne reutilise du depot que ce qui est conforme au
protocole valide ; en particulier il n'appelle NI ``prompts.format_chat``
(qui insere toujours un message systeme) NI ``graphs.build_graph`` (dont les
drapeaux de filtrage de noeuds n'ont pas lieu d'etre ici).

Decisions structurantes, toutes validees avant implementation :

* aucun message systeme : le prompt est un unique tour utilisateur
  ``(instruction + "\\n\\n" + externe).rstrip()`` ;
* tous les tokens sont conserves, puits d'attention compris ; aucun retrait,
  aucune reduction de Kron ;
* trois regions U / E / T, dont les frontieres viennent de la construction du
  prompt et des offsets de tokenisation, jamais des labels d'attaque ;
* aucune selection de couche, de tete ou de bande sur le pouvoir
  discriminant ; aucun apprentissage.
"""

__all__ = ["dataset", "capture", "structural", "provenance"]
