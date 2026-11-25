"""Toutes les decisions prises AVANT de voir un resultat, en un seul endroit.

Rien ici n'est choisi sur les donnees. La cellule declaree pour le vecteur
de Fiedler, la regle de selection des tetes, les tolerances d'appariement
et les tailles de pilote sont fixees ici et enregistrees dans les
metadonnees du run.
"""

from typing import Dict, Optional, Tuple

#: Modele par defaut : petit, instruct, tourne sur CPU. Modifiable.
MODEL_ID = "Qwen/Qwen2.5-0.5B-Instruct"
SEED = 0

#: Taille du pilote : paires (adverse, controle) par jeu de donnees.
N_PAIRS: Dict[str, int] = {"xtram1": 20, "spml": 20}

#: Longueur maximale du prompt formate, systeme inclus. L'attention est en
#: O(T^2) par tete ; au-dela on ecarte et on journalise.
MAX_TOKENS: Dict[str, int] = {"xtram1": 256, "spml": 512}

#: Appariement en longueur : ecart tolere entre adverse et controle, en
#: tokens du prompt COMPLET formate. max(fraction * longueur, minimum).
LENGTH_TOL_FRAC = 0.10
LENGTH_TOL_MIN = 4

#: Configurations du message systeme pour xTRam1 (qui n'en fournit aucun).
#:   ""    -> bloc systeme VIDE  (<|im_start|>system\n<|im_end|>)
#:   None  -> aucun message systeme passe ; le template Qwen INJECTE alors
#:            son systeme par defaut ("You are Qwen, created by Alibaba
#:            Cloud. You are a helpful assistant."), 16 tokens non demandes
#:   str   -> systeme minimal explicite
SYSTEM_CONFIGS: Dict[str, Optional[str]] = {
    "vide": "",
    "defaut_qwen": None,
    "minimal": "You are a helpful assistant.",
}
#: Configuration primaire pour xTRam1 ; les deux autres sont la sensibilite.
PRIMARY_SYSTEM = "vide"

#: Regle de selection des tetes, declaree : TOUTES. Significativite par
#: cellule (couche, tete) avec correction Benjamini-Hochberg au taux
#: ``FDR_Q`` (compte principal) et correction max-statistic FWER (compte
#: secondaire, plus severe). Aucune cellule n'est choisie apres coup.
FDR_Q = 0.05
N_PERM = 20000

#: Metrique primaire des tests par cellule. Les autres metriques de
#: ``spectral-trust`` sont rapportees en descriptif.
PRIMARY_METRIC = "fiedler_value"


def declared_cell(n_layers: int) -> Tuple[int, int]:
    """Cellule (couche, tete) pour le vecteur de Fiedler et les matrices
    d'adjacence : couche mediane, tete 0. Fixee avant tout resultat ;
    arbitraire et assumee comme telle."""
    return (n_layers // 2, 0)


#: Generation pour la mesure comportementale : identique pour tous.
GEN_MAX_NEW_TOKENS = 48
GEN_DO_SAMPLE = False
