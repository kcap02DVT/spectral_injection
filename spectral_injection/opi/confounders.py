"""Plan d'analyse des facteurs de confusion, fixe AVANT le pilote.

Statut : **exploratoire**. Rien ici ne modifie le contraste principal
C4 - C3 ni sa correction familiale. L'objet est de savoir si une difference
observee survit au controle des covariables, et surtout de reperer celles
qui n'y survivent pas.

Les covariables et les regles sont declarees dans ce fichier avant toute
execution, de sorte qu'aucune ne puisse etre ajoutee ou retiree apres avoir
vu les resultats.

----------------------------------------------------------------------
Ce que l'appariement C3/C4 controle deja, et ce qu'il ne controle pas
----------------------------------------------------------------------

Verifie sur le smoke test, 8 lignes sur 8, ecart maximal nul :

    longueur totale du prompt      identique entre C3 et C4
    |U|, |E|, |T|                  identiques
    position de debut de E         identique
    contenu de P_atk et de D_inj   identique

Le contraste principal est donc **deja protege** de la longueur, de la
taille des regions et de la position. Les covariables ci-dessous ne servent
pas a le corriger de ces facteurs, mais a repondre a deux autres questions :

  1. l'effet varie-t-il SELON la longueur, la tache, la famille d'attaque ?
     (modification d'effet, pas confusion)
  2. les contrastes SECONDAIRES -- C4-C2, C4-C1 -- qui eux ne sont pas
     apparies en longueur, sont-ils explicables par la seule longueur ?

======================================================================
CE QUE C4 - C3 N'EST PAS : un effet causal pur de la malveillance
======================================================================

Aucune sortie de ce module, ni du module statistique, ne doit etre lue
comme l'effet causal de l'intention malveillante. C3 et C4 ne different pas
seulement par la malveillance : le bloc remplace differe aussi, et surtout,
sur AU MOINS cinq dimensions simultanees.

  forme syntaxique    F_benign est de la prose declarative ; I_inj est un
                      imperatif. L'echelle de controles menee sur BIPIA a
                      montre que la forme imperative seule portait deja une
                      part substantielle de l'effet mesure.
  destinataire        F_benign s'adresse au lecteur du document ; I_inj
                      s'adresse au modele. Sur BIPIA, c'est l'echelon ou le
                      signal mourait (0,489 contre 0,642).
  changement de tache I_inj demande une tache DIFFERENTE de la tache
                      legitime -- le dataset l'impose, 0 violation sur
                      33 600. C4 contient donc deux instructions en
                      concurrence, C3 une seule.
  contenu lexical     les mots, la ponctuation et la casse different ;
                      l'appariement porte sur le NOMBRE de tokens, pas sur
                      leur identite ni sur leur fragmentation sous-mot.
  coherence semantique F_benign est de la prose tiree d'un autre email du
                      meme corpus, donc du meme registre que l'hote ; I_inj
                      est une consigne hors-sujet.

Une difference significative sur C4 - C3 etablit donc qu'il existe un effet
du remplacement du bloc, **pas** lequel de ces cinq facteurs le produit. La
stratification par ``attack_type`` est le seul levier dont nous disposions
ici pour commencer a les demeler, et elle ne les separe pas non plus : pour
``ignore`` et ``combine``, P_atk porte deja une directive des deux cotes.

Separer ces facteurs exigerait des conditions supplementaires -- un
remplissage imperatif adresse au lecteur, une instruction legitime adressee
au modele -- qui ne font pas partie de ce protocole.
"""

from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

EPS = 1e-12

# ======================================================================
# Covariables declarees
# ======================================================================
#: Covariables d'ECART, laissees NON CENTREES : b0 devient l'effet
#: extrapole a ecart nul. Elles n'ont de sens que pour les contrastes ou les
#: conditions different en longueur, c'est-a-dire C4-C2 et C4-C1. Sur le
#: contraste principal C4-C3 ces ecarts sont nuls par construction, ce qui
#: est verifie et non suppose.
GAP_COVARIATES: Tuple[str, ...] = ("d_T", "d_nE")

#: Covariables de NIVEAU, centrees : elles n'influencent pas b0 et ne font
#: qu'absorber de la variance.
CONTINUOUS: Tuple[str, ...] = (
    "T_C4",            # longueur totale du prompt, condition attaquee
    "nE_C4",           # taille de la region externe
    "nU_C4",           # taille de l'instruction legitime
    "frac_E",          # nE / T : part du prompt occupee par E
    "frac_U",          # nU / T
    "pos_E_start",     # position relative du premier token de E : nU+nT_avant
    "len_I_inj",       # taille en caracteres de l'instruction injectee
    "len_D_inj",       # taille en caracteres des donnees injectees
    "target_tokens_D",  # taille en tokens du bloc remplace
    "frac_injected",   # target_tokens_D / T_C4 : part remplacee
)

#: Categorielles. Servent aux analyses stratifiees, pas a une regression.
CATEGORICAL: Tuple[str, ...] = ("task_type", "attack_type", "injected_task")

#: Regles fixees d'avance.
RULES = {
    "status": "exploratoire : n'affecte pas le contraste principal",
    "min_per_stratum": 4,
    "regression": "moindres carres ordinaires, DEUX blocs de covariables. "
                  "Ecarts entre conditions (d_T, d_nE) laisses NON CENTRES : "
                  "l'intercept est alors l'effet extrapole a ecart nul. "
                  "Niveaux de nuisance CENTRES : ils n'influencent pas "
                  "l'intercept et ne font qu'absorber de la variance.",
    "primary_contrast_note": "sur C4-C3 les ecarts sont nuls par "
                             "construction (verifie 8/8, ecart max 0), donc "
                             "les covariables d'ecart sont automatiquement "
                             "ecartees et le controle ne deplace pas "
                             "l'estimation. C'est attendu, pas un echec.",
    "inference": "permutation par inversion de signe au niveau de "
                 "l'exemple, appliquee aux residus du modele "
                 "covariables-seules (Freedman-Lane)",
    "multiplicity": "Benjamini-Hochberg sur la famille exploratoire, "
                    "separement du max-statistic confirmatoire",
    "no_selection": "aucune covariable n'est ajoutee ou retiree apres "
                    "consultation des resultats",
    "attack_stratification": "obligatoire : pour `ignore` et `combine`, "
                             "P_atk contient lui-meme une directive, donc "
                             "C3 n'y est pas un controle purement benin",
}


def build_covariates(meta) -> Tuple[np.ndarray, List[str]]:
    """``rows.csv`` -> matrice de covariables continues ``[n, p]``.

    Les colonnes absentes sont ignorees et signalees par le nom rendu, de
    sorte qu'un changement de schema ne passe pas inapercu.
    """
    import pandas as pd

    m = meta.copy()
    if {"nE_C4", "T_C4"} <= set(m.columns):
        m["frac_E"] = m.nE_C4 / m.T_C4
    if {"nU_C4", "T_C4"} <= set(m.columns):
        m["frac_U"] = m.nU_C4 / m.T_C4
    if {"nU_C4", "nT_C4"} <= set(m.columns):
        # E commence apres U et apres les tokens d'ouverture du template
        m["pos_E_start"] = m.nU_C4 + 5
    if {"target_tokens_D", "T_C4"} <= set(m.columns):
        m["frac_injected"] = m.target_tokens_D / m.T_C4
    cols = [c for c in CONTINUOUS if c in m.columns]
    return m[cols].to_numpy(dtype=np.float64), cols


def build_gaps(meta, a: str = "C4", b: str = "C3") -> Tuple[np.ndarray, List[str]]:
    """Ecarts de longueur entre deux conditions : ``[n, 2]``.

    Sur C4-C3 ces colonnes sont identiquement nulles ; ``freedman_lane`` les
    ecarte alors d'elle-meme puisque leur ecart-type est nul. C'est le
    comportement voulu : il n'y a rien a extrapoler quand l'appariement est
    exact, et le modele ne doit pas feindre de corriger quelque chose.
    """
    out, names = [], []
    for col, lab in (("T_", "d_T"), ("nE_", "d_nE")):
        ca, cb = f"{col}{a}", f"{col}{b}"
        if ca in meta.columns and cb in meta.columns:
            out.append((meta[ca] - meta[cb]).to_numpy(dtype=np.float64))
            names.append(lab)
    if not out:
        return np.zeros((len(meta), 0)), []
    return np.column_stack(out), names


def freedman_lane(D: np.ndarray, X_zero: Optional[np.ndarray] = None,
                  X_center: Optional[np.ndarray] = None,
                  n_perm: int = 5000, seed: int = 0) -> List[Dict]:
    """Effet extrapole a ecart nul, teste par Freedman-Lane.

    Modele par metrique :

        d = b0 + X_zero . g + X_center . h + e

    **Les deux blocs de covariables ne jouent pas le meme role, et c'est le
    point central.**

    ``X_zero`` reste **non centre**. Il contient les ECARTS entre les deux
    conditions -- difference de longueur totale, difference de taille de E.
    L'ordonnee a l'origine ``b0`` est alors l'effet extrapole a ecart nul,
    c'est-a-dire ce que vaudrait le contraste si les deux conditions avaient
    exactement la meme longueur. C'est la quantite qui repond a
    << l'effet survit-il au controle de la longueur ? >>.

    ``X_center`` est **centre**. Il contient des niveaux de nuisance dont la
    valeur n'est pas d'interet -- longueur absolue du prompt, taille de U.
    Centre, il n'influence pas ``b0`` et ne fait qu'absorber de la variance,
    donc gagner de la puissance.

    Consequence a connaitre : si l'on ne fournit QUE des covariables
    centrees, ``b0`` vaut exactement la moyenne des differences et le
    controle ne deplace pas l'estimation. Sur le contraste principal
    C4 - C3 c'est le comportement attendu, puisque les longueurs y sont
    appariees exactement : il n'existe aucun ecart a extrapoler. Les
    covariables d'ecart n'ont de sens que pour les contrastes SECONDAIRES
    C4 - C2 et C4 - C1, ou les longueurs different.

    Inverser le signe des RESIDUS du modele covariables-seules, plutot que
    des donnees brutes, est ce qui preserve la part expliquee par les
    covariables sous l'hypothese nulle.
    """
    D = np.asarray(D, dtype=np.float64)
    n, m = D.shape
    blocks = []
    if X_zero is not None and np.size(X_zero):
        Z = np.asarray(X_zero, dtype=np.float64).reshape(n, -1)
        blocks += [Z[:, j] for j in range(Z.shape[1]) if Z[:, j].std() > EPS]
    if X_center is not None and np.size(X_center):
        C = np.asarray(X_center, dtype=np.float64).reshape(n, -1)
        blocks += [C[:, j] - C[:, j].mean() for j in range(C.shape[1])
                   if C[:, j].std() > EPS]
    Xc = np.column_stack(blocks) if blocks else np.zeros((n, 0))
    A = np.column_stack([np.ones(n), Xc])
    dof = n - A.shape[1]
    if dof < 2:
        return [{"index": j, "b0": np.nan, "p": np.nan, "dof": dof,
                 "n_covariates": Xc.shape[1]} for j in range(m)]

    pinv = np.linalg.pinv(A)
    fitted = Xc @ np.linalg.pinv(Xc) @ D if Xc.shape[1] else np.zeros_like(D)
    resid = D - fitted

    def b0_stat(Y):
        b = np.einsum("pn,...nm->...pm", pinv, Y)
        e = Y - np.einsum("np,...pm->...nm", A, b)
        s2 = (e ** 2).sum(-2) / dof
        se = np.sqrt(np.maximum(s2 * np.linalg.pinv(A.T @ A)[0, 0], EPS))
        return b[..., 0, :], np.abs(b[..., 0, :]) / se

    b0_obs, t_obs = b0_stat(D)
    rng = np.random.default_rng(seed)
    ge = np.zeros(m)
    done = 0
    while done < n_perm:
        b = min(500, n_perm - done)
        flips = rng.choice([-1.0, 1.0], size=(b, n))
        Y = fitted[None] + flips[:, :, None] * resid[None]
        _, t = b0_stat(Y)
        ge += (t >= t_obs[None] - 1e-12).sum(0)
        done += b
    return [{"index": j, "b0": float(b0_obs[j]), "t": float(t_obs[j]),
             "p": float((ge[j] + 1) / (n_perm + 1)), "dof": int(dof),
             "n_covariates": int(Xc.shape[1])} for j in range(m)]


def stratified_effects(D: np.ndarray, strata: np.ndarray,
                       min_per: int = 4) -> List[Dict]:
    """Effet par strate. Ne teste pas : decrit, et signale les effectifs.

    Une strate sous ``min_per`` exemples est rapportee mais marquee
    ``underpowered`` ; elle n'est pas fusionnee avec une autre, ce qui
    serait un choix fait apres coup.
    """
    out = []
    for s in sorted(set(strata.tolist())):
        idx = np.flatnonzero(strata == s)
        sub = D[idx]
        out.append({"stratum": s, "n": int(idx.size),
                    "underpowered": bool(idx.size < min_per),
                    "median": [float(x) for x in np.median(sub, 0)],
                    "mean": [float(x) for x in sub.mean(0)],
                    "win_rate": [float(x) for x in (sub > 0).mean(0)]})
    return out


def length_sensitivity(D: np.ndarray, length: np.ndarray) -> List[Dict]:
    """Correlation de Spearman entre la difference et une longueur.

    Reponse a la question << l'effet depend-il de la taille du prompt ? >>.
    C'est une modification d'effet, pas une confusion : sur le contraste
    principal les longueurs sont appariees exactement.
    """
    from scipy.stats import spearmanr

    out = []
    for j in range(D.shape[1]):
        if np.std(D[:, j]) < EPS or np.std(length) < EPS:
            out.append({"index": j, "rho": np.nan, "p": np.nan})
            continue
        r, p = spearmanr(D[:, j], length)
        out.append({"index": j, "rho": float(r), "p": float(p)})
    return out
