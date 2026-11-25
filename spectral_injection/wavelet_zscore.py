"""Standardisation intra-prompt multi-echelles : ondelettes + z-score local.

Repond aux trois ecueils mesures dans les campagnes precedentes :

    variance inter-prompts   l'energie absolue vaut 0,88 en apparie et 0,57
                             en pooled. Le z-score est calcule a l'interieur
                             du prompt, donc le niveau de fond propre au
                             document disparait par construction : il n'y a
                             plus rien a comparer entre documents.
    taille inconnue          l'approche K-means devait figer la taille de
                             l'anomalie et a heurte un plafond arithmetique.
                             Ici on balaye les echelles et on prend le
                             maximum : aucune taille n'est fixee d'avance.
    maximum absolu           l'ondelette brute echouait en aveugle parce que
                             l'injection n'est pas le maximum d'energie. Le
                             z-score ne cherche plus un maximum d'energie
                             mais un ecart a la distribution des fenetres du
                             meme prompt, ce qui n'est pas la meme cible.

Trois precautions, implementees et non laissees implicites :

1.  **L'echelle n'est pas une largeur en tokens.** Dans le SGWT, s est une
    dilatation *spectrale* : le noyau g(s lambda) se concentre autour de
    lambda = 1/s. L'etendue spatiale de psi_{s,i} en decoule, mais aucun
    calcul ne garantit que s = 5 corresponde a 5 tokens. ``spatial_extent``
    mesure l'etalement reel de chaque echelle, en tokens, pour que la
    correspondance soit verifiee au lieu d'etre supposee. Le parametre
    ``window_sizes`` permet de decoupler la largeur de fenetre de l'echelle
    spectrale ; par defaut il suit l'enonce (fenetre = s).

2.  **Les fenetres glissantes se recouvrent**, donc les T-s+1 z-scores d'une
    echelle sont fortement correles et leur maximum n'est pas distribue
    comme le maximum de T-s+1 normales independantes. Un z de 4 n'est pas un
    evenement a 3e-5. ``surrogate_pvalue`` calibre le maximum par surrogates
    a spectre de puissance conserve (randomisation de phase) : on garde
    l'autocorrelation du profil et on detruit les bosses localisees, donc la
    p-valeur repond exactement a << cette concentration depasse-t-elle ce que
    l'autocorrelation du profil explique deja >>.

3.  **Le z-score est en leave-one-out**, comme demande : chaque fenetre est
    standardisee contre les *autres* fenetres, jamais contre un ensemble qui
    la contient. Sans ca une fenetre tres anormale gonfle le sigma qui la
    juge et se masque elle-meme. Forme fermee, O(m), pas de boucle.

Graphe d'entree : G1-Kron par defaut, puits deja reduits par complement de
Schur -- la tache 1 a montre que la suppression seche fait s'effondrer
lambda_2 de 72 %, alors que Kron preserve les resistances effectives. La
sparsification est q90 par defaut et non ``dense`` : sur l'attention dense,
mesure sur le cache, l'energie d'ondelette est uniforme a 1/T pres
(IPR = 0,0041 = 1/244) et il n'y a rien a localiser.

    python -m spectral_injection.wavelet_zscore --selftest
    python -m spectral_injection.wavelet_zscore cache/attn/MODELE/SPLIT \\
        --layer-stride 8 --head-stride 7 --per-head
"""

import argparse
import os
import sys
from typing import Dict, List, Optional, Sequence, Tuple, Union

import numpy as np

from .prompts import ROLE_INJECTION
from .wavelets import (KERNELS, LMAX_SYM, auto_order, chebyshev_apply,
                       chebyshev_coeffs, graph_from_attention,
                       wavelet_node_energy)

EPS = 1e-12

#: Echelles spectrales. Mesurees, pas supposees : sur ce cache le signal est
#: a s = 0,5-1,7, soit un etalement spatial inferieur au token -- le
#: voisinage immediat de chaque noeud. Le jeu grossier initialement essaye,
#: (5, 10, 15, 20, 30), correspond a 4-20 tokens d'etalement sur un graphe
#: d'attention reel et passe au-dessus du signal : la campagne complete y
#: rend 0,48 d'AUROC, contre 0,75 aux echelles fines.
SCALES = (0.5, 1.0, 1.71, 3.0, 5.0)

#: Largeurs de fenetre, en tokens, DECOUPLEES des echelles. L'enonce
#: initial posait fenetre = s, ce qui est intenable des que s < 1. Les deux
#: parametres repondent a deux questions distinctes : l'echelle fixe la
#: finesse de la structure sondee, la largeur fixe l'etendue sur laquelle on
#: agrege. Le produit croise des deux est balaye.
WIDTHS = (8, 11, 15, 20, 30)

#: Hypothese PRE-ENREGISTREE, arretee le 3 novembre 2026 sur le split train
#: de Qwen2.5-0.5B-Instruct, AVANT toute mise en cache du split test.
#:
#: Les categories d'attaque de train et de test sont disjointes dans BIPIA
#: (15 de chaque cote), donc une lecture sur test est hors distribution par
#: construction. La condition pour que ce soit un resultat et non une
#: exploration : lire UNIQUEMENT la ligne ci-dessous, sans consulter le
#: maximum du nouveau run. Un seul candidat, donc aucune correction pour
#: selection a appliquer.
#:
#: La cellule a ete reperee par des balayages successifs sur les 37 paires de
#: train ; sa valeur de train est donc optimiste et n'est pas l'enjeu.
#: L'enjeu est le signe et l'ordre de grandeur sur test.
PREREGISTRE = {
    "model": "Qwen2.5-0.5B-Instruct",
    "layer": 8,
    "head": 7,
    "reduction": "kron",
    "sparse": "q90",
    "kernel": "mexican_hat",
    "tail": "both",
    "rule": "argmax|z|",            # et non argmin(p), mesuree inferieure
    "scales": (0.5, 1.0, 1.71, 3.0, 5.0),
    "widths": (8, 11, 15, 20, 30),
    "train_auroc_paired": 0.676,
    "train_auroc_pooled": 0.570,
    "train_n_pairs": 37,
    # Ce qui compte comme confirmation, fixe d'avance :
    #   apparie > 0,5 avec un intervalle binomial excluant 0,5  -> signal
    #   apparie autour de 0,5                                   -> la cellule
    #                                                   surapprenait le bruit
}


# --------------------------------------------------------------------------
# Tache 1 : profil d'ondelettes multi-echelles
# --------------------------------------------------------------------------
def multiscale_wavelet_profile(L, scales: Sequence[float] = SCALES,
                               kernel: str = "mexican_hat",
                               order: Optional[int] = None,
                               probes: Optional[int] = None,
                               seed: int = 0) -> np.ndarray:
    """``(len(scales), T)`` : energie d'ondelette par noeud et par echelle.

    E[j, i] = || psi_{s_j, i} ||^2 = [g(s_j L)^2]_{ii}, obtenu en appliquant
    le polynome de Tchebychev a l'identite. Aucun signal exterieur, aucune
    etiquette : seul le graphe entre.

    Le degre du polynome suit l'echelle (``auto_order``) : a degre fixe 30 le
    noyau est faux au-dela de s ~ 10, ce que le selftest de ``wavelets``
    avait mesure a 4e-2 d'erreur sur s = 20. Ici les echelles vont jusqu'a
    30, donc le degre monte jusqu'a 190 -- c'est le prix de l'exactitude a
    grande echelle, et il reste lineaire en s.
    """
    return wavelet_node_energy(L, scales, kernel, order, LMAX_SYM, probes, seed)


def spatial_extent(L, scales: Sequence[float] = SCALES,
                   kernel: str = "mexican_hat",
                   sample: int = 24, seed: int = 0) -> np.ndarray:
    """Etalement spatial reel de chaque echelle, en tokens : ``(len(scales),)``.

    Pour un echantillon de noeuds i, on calcule psi_{s,i} puis la distance
    moyenne ponderee sum_j psi^2(j) |i - j| / sum_j psi^2(j). C'est la
    quantite qui dit si << echelle 5 >> veut dire cinq tokens ou cinquante.
    A verifier avant d'interpreter un balayage d'echelles comme un balayage
    de tailles d'injection.

    L'etalement depend du graphe, pas seulement de l'echelle : il est fixe
    par la densite spectrale. Mesure sur une chaine de 200 noeuds, s = 30
    donne 3,9 tokens ; sur un graphe d'attention reel (Kron, q90, n = 244) la
    meme echelle en donne 20,0. Un jeu d'echelles calibre sur un graphe ne
    se transporte donc pas tel quel sur un autre.

    Deux regimes degeneres sont rendus en NaN plutot qu'en nombre trompeur :
    quand l'operateur s'evanouit (g(s lambda) passe sous le plancher
    spectral du graphe, c'est-a-dire s >> 1/lambda_2, et psi tend vers zero),
    et quand l'etalement sature a la taille du graphe.
    """
    n = L.shape[0]
    rng = np.random.default_rng(seed)
    idx = rng.choice(n, size=min(sample, n), replace=False)
    out = np.empty(len(scales))
    X = np.zeros((n, idx.size))
    X[idx, np.arange(idx.size)] = 1.0
    d = np.abs(np.arange(n)[:, None] - idx[None, :]).astype(float)
    for k, s in enumerate(scales):
        c = chebyshev_coeffs(kernel, float(s), None, LMAX_SYM)
        P = chebyshev_apply(L, X, c, LMAX_SYM) ** 2            # [n, sample]
        tot = P.sum(0)
        # psi evanoui : l'energie totale est sous le bruit numerique, le
        # rapport qui suit serait un quotient de zeros.
        if np.median(tot) < 1e-10:
            out[k] = np.nan
            continue
        out[k] = float(np.mean((P * d).sum(0) / np.maximum(tot, EPS)))
    return out


# --------------------------------------------------------------------------
# Tache 2 : fenetre glissante et z-score local en leave-one-out
# --------------------------------------------------------------------------
def window_means(x: np.ndarray, width: int) -> np.ndarray:
    """Moyennes glissantes sur fenetres contigues : ``(T - width + 1,)``.

    Somme cumulee, donc O(T) quelle que soit la largeur.
    """
    w = int(max(1, min(width, x.size)))
    c = np.concatenate([[0.0], np.cumsum(x)])
    return (c[w:] - c[:-w]) / w


def loo_zscore(v: np.ndarray) -> np.ndarray:
    """Z-score leave-one-out : chaque valeur contre *les autres*.

        mu_i    = (S1 - v_i) / (m - 1)
        sigma_i^2 = (S2 - v_i^2 - (m-1) mu_i^2) / (m - 2)
        z_i     = (v_i - mu_i) / sigma_i

    Forme fermee exacte, O(m). Le leave-one-out n'est pas un detail : avec
    m ~ 230 fenetres l'ecart sur mu est faible, mais une fenetre tres
    anormale contribue a sigma et se masque partiellement elle-meme, ce qui
    biaise precisement le cas qu'on cherche a detecter.
    """
    v = np.asarray(v, dtype=np.float64)
    m = v.size
    if m < 4:
        return np.zeros(m)
    s1, s2 = v.sum(), (v ** 2).sum()
    mu = (s1 - v) / (m - 1)
    var = (s2 - v ** 2 - (m - 1) * mu ** 2) / (m - 2)
    sd = np.sqrt(np.maximum(var, 0.0))
    return np.where(sd > EPS, (v - mu) / np.maximum(sd, EPS), 0.0)


def local_zscore_windows(energy: np.ndarray,
                         scales: Sequence[float] = SCALES,
                         window_sizes: Optional[Sequence[int]] = None
                         ) -> Tuple[List[np.ndarray], np.ndarray]:
    """Z-scores locaux, echelle par echelle.

    Rend ``(liste, matrice, cles)`` : un vecteur de z par couple
    (echelle, largeur), la meme information en matrice completee de NaN
    pour un argmax global, et la liste des couples dans l'ordre des lignes.

    Le produit croise remplace l'appariement fenetre = s de l'enonce. Deux
    raisons, l'une formelle et l'autre mesuree : s < 1 ne definit aucune
    largeur entiere, et le signal se trouve a s = 0,5-1,7 avec des largeurs
    utiles de 8 a 30 tokens -- les deux parametres ne sont pas la meme
    quantite et ne peuvent pas etre lies.
    """
    E = np.asarray(energy, dtype=np.float64)
    if E.ndim != 2 or E.shape[0] != len(scales):
        raise ValueError(f"attendu ({len(scales)}, T), recu {E.shape}")
    widths = list(WIDTHS) if window_sizes is None else [int(w) for w in window_sizes]
    zs, keys = [], []
    for k in range(len(scales)):
        for w in widths:
            zs.append(loo_zscore(window_means(E[k], w)))
            keys.append((k, int(w)))
    m = max(z.size for z in zs)
    Z = np.full((len(zs), m), np.nan)
    for i, z in enumerate(zs):
        Z[i, :z.size] = z
    return zs, Z, keys


# --------------------------------------------------------------------------
# Calibration : le maximum sur des fenetres recouvrantes n'est pas gaussien
# --------------------------------------------------------------------------
def phase_surrogate(x: np.ndarray, rng) -> np.ndarray:
    """Surrogate a spectre de puissance identique, phases randomisees.

    On garde |FFT(x)| et on tire les phases au hasard : l'autocorrelation du
    profil est preservee, toute bosse localisee est detruite. C'est le null
    qui correspond a la question posee -- << y a-t-il une concentration que
    l'autocorrelation n'explique pas >> -- et non a << le profil est-il du
    bruit blanc >>, qui serait un null trop facile.
    """
    F = np.fft.rfft(x - x.mean())
    ph = rng.uniform(0.0, 2.0 * np.pi, F.size)
    ph[0] = 0.0
    if x.size % 2 == 0:
        ph[-1] = 0.0
    out = np.fft.irfft(np.abs(F) * np.exp(1j * ph), n=x.size)
    return out + x.mean()


def surrogate_pvalue(energy: np.ndarray, scales: Sequence[float] = SCALES,
                     window_sizes: Optional[Sequence[int]] = None,
                     n_surrogates: int = 200, seed: int = 0,
                     tail: str = "both") -> Dict[str, float]:
    """p-valeur intra-prompt du maximum de z, par surrogates de phase.

    Rend le z maximum observe, le z maximum moyen sous surrogate, et

        p = (#{max_z surrogate >= max_z observe} + 1) / (n + 1)

    C'est une calibration *par prompt*, sans aucun apprentissage et sans
    reference exterieure : elle repond a la question que le z brut ne peut
    pas trancher, puisque le maximum sur des centaines de fenetres
    recouvrantes est eleve meme sans anomalie.
    """
    def stat(Zm):
        if tail == "upper":
            return float(np.nanmax(Zm))
        if tail == "lower":
            return float(-np.nanmin(Zm))
        return float(np.nanmax(np.abs(Zm)))

    rng = np.random.default_rng(seed)
    E = np.asarray(energy, dtype=np.float64)
    _, Z, _ = local_zscore_windows(E, scales, window_sizes)
    obs = stat(Z)
    nulls = np.empty(n_surrogates)
    for b in range(n_surrogates):
        Es = np.stack([phase_surrogate(E[k], rng) for k in range(E.shape[0])])
        _, Zs, _ = local_zscore_windows(Es, scales, window_sizes)
        nulls[b] = stat(Zs)
    return {"z_max": obs, "z_max_null_mean": float(nulls.mean()),
            "z_max_null_q95": float(np.quantile(nulls, 0.95)),
            "p_surrogate": float((np.sum(nulls >= obs) + 1) / (n_surrogates + 1))}


# --------------------------------------------------------------------------
# Tache 3 : la regle de decision aveugle
# --------------------------------------------------------------------------
def blind_decision(Z: np.ndarray, keys: Sequence[Tuple[int, int]],
                   scales: Sequence[float] = SCALES,
                   positions: Optional[np.ndarray] = None,
                   top_k: int = 5, tail: str = "both") -> Dict[str, object]:
    """Couple (echelle, largeur) et fenetre les plus anormaux. Aucune etiquette.

    ``keys`` est la troisieme sortie de ``local_zscore_windows`` : il donne,
    pour chaque ligne de ``Z``, l'indice d'echelle et la largeur de fenetre.

    ``tail`` est le parametre qu'il ne faut pas laisser par defaut sans y
    penser :

        upper  on cherche un pic d'energie  (max z)
        lower  on cherche un creux          (min z)
        both   on cherche un ecart          (max |z|)   <- defaut

    La mesure tranche : sur un span plante, l'energie d'ondelette passe-bande
    est 30 a 100 fois *plus faible* que l'environnement (z = -4,95 contre un
    max positif de +0,81). Une ondelette passe-bande repond aux
    heterogeneites, et un span fortement lie a lui-meme est une zone
    localement homogene. Chercher un maximum revient a regarder la mauvaise
    queue. ``both`` est le defaut parce qu'une regle aveugle ne peut pas
    supposer connaitre le signe ; ``lower`` est le choix a faire si la
    direction se confirme sur un split independant.

    Avertissement sur l'agregation. Prendre l'argmax de |z| *a travers* les
    couples favorise mecaniquement la plus petite largeur : moins de tokens
    moyennes, plus de variance, donc des z plus extremes. La campagne l'a
    montre -- la plus petite echelle etait choisie dans 74 % des cas. Pour
    une comparaison honnete entre couples il faut passer par une p-valeur
    calibree par couple (``surrogate_pvalue`` par ligne) et prendre le
    minimum, et non comparer des z bruts.
    """
    if tail not in ("upper", "lower", "both"):
        raise ValueError(f"tail inconnu : {tail}")
    ok = np.isfinite(Z)
    signed = np.where(ok, Z, 0.0)
    if tail == "upper":
        rank = np.where(ok, Z, -np.inf)
    elif tail == "lower":
        rank = np.where(ok, -Z, -np.inf)
    else:
        rank = np.where(ok, np.abs(Z), -np.inf)
    flat_rank, flat_signed = rank.ravel(), signed.ravel()
    order = np.argsort(-flat_rank)[:max(top_k, 1)]
    cand = []
    for f in order:
        if not np.isfinite(flat_rank[f]):
            continue
        row, j = divmod(int(f), Z.shape[1])
        k, w = keys[row]
        lo, hi = j, j + w - 1
        if positions is not None:
            lo = int(positions[min(lo, positions.size - 1)])
            hi = int(positions[min(hi, positions.size - 1)])
        cand.append({"z": float(flat_signed[f]), "score": float(flat_rank[f]),
                     "scale": float(scales[k]), "width": int(w),
                     "start": int(lo), "end": int(hi)})
    if not cand:
        return {"z_max": float("nan"), "score": float("nan"),
                "scale": float("nan"), "width": 0, "start": -1, "end": -1,
                "tail_used": tail, "top": []}
    b = cand[0]
    return {"z_max": b["z"], "score": b["score"], "scale": b["scale"],
            "width": b["width"], "start": b["start"], "end": b["end"],
            "tail_used": "lower" if b["z"] < 0 else "upper", "top": cand}


def calibrate_scales(L, target_extents: Sequence[float] = (3, 6, 11, 18, 30),
                     kernel: str = "mexican_hat", lo: float = 1.0,
                     hi: float = 500.0, iters: int = 18,
                     sample: int = 12, seed: int = 0) -> np.ndarray:
    """Echelles spectrales dont l'etalement *mesure* vaut les cibles voulues.

    Repond au probleme signale par ``spatial_extent`` : s = 5 ne vaut pas
    cinq tokens mais 1,5, et s = 30 en vaut 3,1. L'etalement croit tres
    lentement en s, donc balayer s de 5 a 30 ne balaye pratiquement aucune
    taille. Cette fonction inverse la relation par bissection sur
    l'etalement mesure, de sorte que les echelles rendues couvrent bien la
    plage de tailles demandee, en tokens.

    ``target_extents`` vaut par defaut 3 a 30 tokens : le span median du
    cache fait 11 tokens, avec un minimum de 5 et un maximum de 23.
    Monotonie : l'etalement croit avec s, ce qui rend la bissection valide ;
    si la cible est hors de [etendue(lo), etendue(hi)], la borne est rendue.
    """
    # La borne haute n'est pas arbitraire : au-dela de s ~ 1/lambda_2 le
    # noyau passe sous le plancher spectral et psi s'evanouit. On mesure donc
    # d'abord ce que la borne haute donne, et toute cible au-dela est rendue
    # en NaN plutot qu'en echelle inutilisable.
    e_hi = float(spatial_extent(L, [hi], kernel, sample, seed)[0])
    out = []
    for target in target_extents:
        if not np.isfinite(e_hi) or e_hi < target:
            out.append(np.nan)                  # etalement hors d'atteinte
            continue
        a, b = lo, hi
        for _ in range(iters):
            mid = np.sqrt(a * b)                       # bissection en log
            e = float(spatial_extent(L, [mid], kernel, sample, seed)[0])
            if not np.isfinite(e) or e < target:
                a = mid
            else:
                b = mid
        out.append(float(np.sqrt(a * b)))
    return np.array(out)



# --------------------------------------------------------------------------
# Regle aveugle calibree : argmin(p) et non argmax(|z|)
# --------------------------------------------------------------------------
def _tail_stat(z: np.ndarray, tail: str) -> float:
    """Statistique d'un vecteur de z selon la direction cherchee.

    Toujours rendue positive-grande-est-anormal, pour que les couples soient
    comparables et que le null se lise dans le meme sens.
    """
    if z.size == 0:
        return float("nan")
    if tail == "upper":
        return float(np.nanmax(z))
    if tail == "lower":
        return float(-np.nanmin(z))
    return float(np.nanmax(np.abs(z)))


def _tail_argmax(z: np.ndarray, tail: str) -> int:
    """Indice de fenetre qui realise la statistique."""
    if tail == "upper":
        return int(np.nanargmax(z))
    if tail == "lower":
        return int(np.nanargmin(z))
    return int(np.nanargmax(np.abs(z)))


def _couple_stats(E: np.ndarray, widths: Sequence[int], tail: str
                  ) -> np.ndarray:
    """Statistique de chaque couple (echelle, largeur) : ``(n_couples,)``."""
    out = []
    for k in range(E.shape[0]):
        for w in widths:
            out.append(_tail_stat(loo_zscore(window_means(E[k], int(w))), tail))
    return np.asarray(out, dtype=np.float64)


def calibrated_blind_decision(
    energy: np.ndarray, scales: Sequence[float] = SCALES,
    window_sizes: Optional[Sequence[int]] = None,
    positions: Optional[np.ndarray] = None, n_surrogates: int = 200,
    tail: str = "both", seed: int = 0, top_k: int = 5
) -> Dict[str, object]:
    """Regle aveugle par p-valeur calibree couple par couple.

    Le probleme que ca corrige. ``argmax |z|`` a travers les couples choisit
    la plus petite largeur dans 69 % des cas, parce qu'une fenetre de 8
    tokens moyenne moins de valeurs, garde plus de variance, et produit donc
    des z plus extremes a structure egale. Les z de deux largeurs
    differentes ne sont pas sur la meme echelle et leur comparaison directe
    n'a pas de sens.

    Ce que ca fait a la place. Chaque couple (s, w) recoit son *propre* null
    par randomisation de phase -- meme spectre de puissance, donc meme
    autocorrelation, bosses localisees detruites -- et la statistique
    observee est convertie en p-valeur dans ce null. Le vainqueur est le
    couple de p minimale : le plus improbable selon sa propre reference, non
    le plus extreme dans l'absolu. La comparaison redevient licite parce que
    chaque couple est juge contre lui-meme.

    Les surrogates sont tires une fois par echelle et partages par toutes
    les largeurs de cette echelle : c'est voulu, ca preserve la dependance
    entre largeurs d'une meme echelle au lieu de la casser.

    Egalites. Beaucoup de couples atteignent le plancher 1/(n+1) -- avec 200
    surrogates, 0,00498. Le departage se fait sur la statistique
    standardisee par le null,

        z_std = (obs - moyenne du null) / ecart-type du null,

    qui mesure la meme improbabilite mais continument, donc sans plancher.
    C'est aussi elle qui sert de score du prompt pour l'AUROC : une
    p-valeur saturee produirait des rangs ex aequo en masse et degraderait
    l'AUROC pour une raison purement numerique.
    """
    if tail not in ("upper", "lower", "both"):
        raise ValueError(f"tail inconnu : {tail}")
    E = np.asarray(energy, dtype=np.float64)
    widths = list(WIDTHS) if window_sizes is None else [int(w) for w in window_sizes]
    zs, _, keys = local_zscore_windows(E, scales, widths)
    obs = np.array([_tail_stat(z, tail) for z in zs])

    rng = np.random.default_rng(seed)
    nulls = np.empty((n_surrogates, obs.size))
    for b in range(n_surrogates):
        Es = np.stack([phase_surrogate(E[k], rng) for k in range(E.shape[0])])
        nulls[b] = _couple_stats(Es, widths, tail)

    ge = np.sum(nulls >= obs[None, :], axis=0)
    p = (ge + 1.0) / (n_surrogates + 1.0)
    mu = nulls.mean(0)
    sd = nulls.std(0, ddof=1)
    z_std = (obs - mu) / np.maximum(sd, EPS)

    # p croissante d'abord, z_std decroissante pour departager.
    order = np.lexsort((-z_std, p))
    cand = []
    for row in order[:max(top_k, 1)]:
        k, w = keys[row]
        j = _tail_argmax(zs[row], tail)
        lo, hi = j, j + int(w) - 1
        if positions is not None:
            lo = int(positions[min(lo, positions.size - 1)])
            hi = int(positions[min(hi, positions.size - 1)])
        cand.append({"p": float(p[row]), "z_std": float(z_std[row]),
                     "z_obs": float(zs[row][j]), "stat": float(obs[row]),
                     "null_mean": float(mu[row]), "null_sd": float(sd[row]),
                     "scale": float(scales[k]), "width": int(w),
                     "start": int(lo), "end": int(hi)})
    if not cand:
        return {"p_cal": float("nan"), "score_zstd": float("nan"),
                "score_logp": float("nan"), "z_max": float("nan"),
                "scale": float("nan"), "width": 0, "start": -1, "end": -1,
                "n_tied": 0, "top": []}
    b0 = cand[0]
    return {"p_cal": b0["p"], "score_zstd": b0["z_std"],
            "score_logp": float(-np.log10(max(b0["p"], EPS))),
            "z_max": b0["z_obs"], "stat": b0["stat"],
            "null_mean": b0["null_mean"], "null_sd": b0["null_sd"],
            "scale": b0["scale"], "width": b0["width"],
            "start": b0["start"], "end": b0["end"],
            "n_tied": int(np.sum(p <= b0["p"] + EPS)), "top": cand}


# --------------------------------------------------------------------------
# Pipeline complet, d'un tenseur d'attention a un score
# --------------------------------------------------------------------------
def score_prompt(attn_layer: np.ndarray, roles: np.ndarray,
                 scales: Sequence[float] = SCALES,
                 head: Optional[int] = None, reduction: str = "kron",
                 sparse: str = "q90", kernel: str = "mexican_hat",
                 window_sizes: Optional[Sequence[int]] = None,
                 n_surrogates: int = 0, top_k: int = 5,
                 tail: str = "both", calibrated: bool = False
                 ) -> Dict[str, object]:
    """Les trois taches enchainees sur une couche d'attention.

    ``calibrated=False`` par defaut, et c'est une decision prise sur
    mesure et non par commodite. La regle argmin(p) corrige bien le biais de
    largeur -- la part de w=8 tombe de 71,6 % a 55,4 % -- mais elle degrade
    la detection la ou le signal se trouve : a la cellule couche 8 tete 7,
    l'AUROC pooled passe de 0,570 a 0,508 et l'apparie de 0,676 a 0,568.

    La lecture est que le biais n'etait pas un defaut. Le signal vit a un
    etalement sub-token, donc la fenetre la plus etroite est l'informative,
    et calibrer chaque couple contre son propre null repond a << ce z est-il
    improbable pour cette taille >> alors que la question utile est << ce z
    est-il improbable tout court >>. L'identite du couple porte de
    l'information, l'egaliser la detruit.

    La regle calibree reste disponible (``calibrated=True``, ou
    ``--calibrated-rule``) et ses sorties restent rendues sous ``p_cal`` et
    ``score_zstd`` : le resultat negatif est reproductible, il n'est pas
    efface.
    """
    L, r, pos = graph_from_attention(attn_layer, roles, reduction,
                                     head=head, sparse=sparse)
    E = multiscale_wavelet_profile(L, scales, kernel)
    _, Z, keys = local_zscore_windows(E, scales, window_sizes)
    raw = blind_decision(Z, keys, scales, pos, top_k, tail)
    out = dict(raw)
    out["score_raw"] = raw["score"]
    out["start_raw"], out["end_raw"] = raw["start"], raw["end"]
    if calibrated and n_surrogates:
        cal = calibrated_blind_decision(E, scales, window_sizes, pos,
                                        n_surrogates, tail, 0, top_k)
        out.update(cal)
    elif n_surrogates:
        out.update(surrogate_pvalue(E, scales, window_sizes, n_surrogates,
                                    0, tail))
    out["n_nodes"] = int(L.shape[0])
    out["_roles"] = r
    out["_pos"] = pos
    out["_Z"] = Z
    return out


def span_iou(start: int, end: int, roles: np.ndarray,
             positions: np.ndarray) -> float:
    """Intersection sur union entre la fenetre predite et le span reel."""
    inj = positions[roles == ROLE_INJECTION]
    if inj.size == 0:
        return float("nan")
    win = np.arange(start, end + 1)
    return float(np.intersect1d(win, inj).size / max(np.union1d(win, inj).size, 1))


# --------------------------------------------------------------------------
# Verification
# --------------------------------------------------------------------------
def selftest(seed: int = 0) -> int:
    """Le z-score en forme fermee contre la definition naive, et la chaine
    complete sur un graphe ou l'anomalie est plantee a un endroit connu."""
    rng = np.random.default_rng(seed)
    ok = True

    # 1. leave-one-out : forme fermee contre boucle explicite
    v = rng.standard_normal(60) * 3.0 + 7.0
    v[12] += 20.0
    fast = loo_zscore(v)
    slow = np.array([(v[i] - np.delete(v, i).mean())
                     / np.delete(v, i).std(ddof=1) for i in range(v.size)])
    err = np.abs(fast - slow).max()
    print(f"z leave-one-out, forme fermee vs boucle : erreur max {err:.2e}")
    ok &= err < 1e-10

    # Et la demonstration que le leave-one-out n'est pas cosmetique :
    naive = (v - v.mean()) / v.std(ddof=1)
    print(f"  pic a l'indice 12 : LOO z = {fast[12]:.2f}, "
          f"z naif = {naive[12]:.2f}  "
          f"(le naif se masque de {100*(1-naive[12]/fast[12]):.0f} %)")

    # 2. chaine complete : bloc dense plante sur 12 tokens d'une chaine
    n, lo, hi = 200, 80, 91
    P = np.zeros((n, n))
    i = np.arange(n - 1)
    P[i, i + 1] = P[i + 1, i] = 1.0           # chaine : structure de texte
    P[lo:hi + 1, lo:hi + 1] += 3.0            # le span, dense sur lui-meme
    np.fill_diagonal(P, 0.0)
    from .sinks import laplacian_sym
    L = laplacian_sym(P)
    E = multiscale_wavelet_profile(L, SCALES)
    zs, Z, keys = local_zscore_windows(E, SCALES)
    print(f"anomalie plantee en [{lo}, {hi}] :")
    for t in ("upper", "lower", "both"):
        dec = blind_decision(Z, keys, SCALES, top_k=3, tail=t)
        a = set(range(dec["start"], dec["end"] + 1))
        b = set(range(lo, hi + 1))
        iou = len(a & b) / max(len(a | b), 1)
        print(f"  tail={t:<6} -> [{dec['start']:>3}, {dec['end']:>3}] "
              f"s={dec['scale']:>5.2f} w={dec['width']:>3}  "
              f"z={dec['z_max']:>6.2f}  IoU={iou:.2f}")
        if t == "both":
            ok &= len(a & b) > 0

    ext = spatial_extent(L, SCALES)
    print("etalement spatial par echelle, SUR CETTE CHAINE (tokens) --")
    print("  propre au graphe : un graphe d'attention reel donne ~5x plus")
    for s, e in zip(SCALES, ext):
        print(f"  s = {s:>4.0f} -> {e:>6.1f} tokens")

    for t in ("upper", "both"):
        cal = surrogate_pvalue(E, SCALES, n_surrogates=100, tail=t)
        print(f"calibration tail={t:<6} : stat = {cal['z_max']:.2f}, "
              f"null moyen {cal['z_max_null_mean']:.2f}, "
              f"q95 {cal['z_max_null_q95']:.2f}, p = {cal['p_surrogate']:.3f}")

    print("echelles recalibrees pour couvrir 3 a 30 tokens d'etalement :")
    cs = calibrate_scales(L, (3, 6, 11, 18, 30), sample=8)
    ext2 = spatial_extent(L, cs, sample=8)
    for t, sc, e in zip((3, 6, 11, 18, 30), cs, ext2):
        print(f"  cible {t:>3} tokens -> s = {sc:>12.1f}  (etalement "
              f"mesure {e:>5.1f})")

    print("\n" + ("OK" if ok else "ECHEC"))
    return 0 if ok else 1


# --------------------------------------------------------------------------
# Campagne sur le cache
# --------------------------------------------------------------------------
def run(cache_dir: str, layer_stride: int = 8, head_stride: int = 7,
        per_head: bool = False, scales: Sequence[float] = SCALES,
        reduction: str = "kron", sparse: str = "q90",
        kernel: str = "mexican_hat", n_surrogates: int = 0,
        top_k: int = 5, max_pairs: Optional[int] = None,
        tail: str = "both", calibrate: bool = False,
        window_sizes: Optional[Sequence[int]] = None,
        calibrated_rule: bool = False) -> List[Dict]:
    from .attn_cache import iter_cache
    import time

    rows: List[Dict] = []
    n = 0
    t0 = time.time()
    widths = list(window_sizes or WIDTHS)   # largeurs fixes, independantes :
    # si les echelles sont recalibrees par prompt, les fenetres doivent rester
    # comparables d'un prompt a l'autre, sinon le z n'est plus la meme
    # statistique d'un document au suivant.
    for att_i, att_c, roles_i, roles_c, meta in iter_cache(cache_dir):
        n += 1
        heads = (list(range(0, att_i.shape[1], head_stride)) if per_head
                 else [None])
        for l in range(0, att_i.shape[0], layer_stride):
            for h in heads:
                sc = scales
                if calibrate:
                    Lref, _, _ = graph_from_attention(att_i[l], roles_i,
                                                      reduction, head=h,
                                                      sparse=sparse)
                    sc = calibrate_scales(Lref, (3, 6, 11, 18, 30), kernel,
                                          sample=8)
                for cond, att, roles in (("injected", att_i, roles_i),
                                         ("control", att_c, roles_c)):
                    d = score_prompt(att[l], roles, sc, h, reduction,
                                     sparse, kernel, widths, n_surrogates,
                                     top_k, tail, calibrated_rule)
                    r, pos = d.pop("_roles"), d.pop("_pos")
                    d.pop("_Z", None)
                    iou = span_iou(d["start"], d["end"], r, pos)
                    # la vraie cible est-elle dans les top_k fenetres ?
                    inj = pos[r == ROLE_INJECTION]
                    hit_k = 0.0
                    for c in d["top"]:
                        w = np.arange(c["start"], c["end"] + 1)
                        if inj.size and np.intersect1d(w, inj).size:
                            hit_k = 1.0
                            break
                    iou_raw = span_iou(d["start_raw"], d["end_raw"], r, pos)
                    rows.append({
                        "pair": int(meta["pair"]), "layer": int(l),
                        "head": -1 if h is None else int(h),
                        "condition": cond, "position": meta["position"],
                        "z_max": d["z_max"], "scale": d["scale"],
                        "width": d["width"],
                        "start": d["start"], "end": d["end"],
                        "iou": iou, "hit_topk": hit_k,
                        # regle calibree
                        "p_cal": d.get("p_cal", float("nan")),
                        "score_zstd": d.get("score_zstd", float("nan")),
                        "score_logp": d.get("score_logp", float("nan")),
                        "n_tied": d.get("n_tied", 0),
                        # ancienne regle, memes graphes, pour comparaison
                        "score_raw": d["score_raw"],
                        "iou_raw": iou_raw,
                        "n_nodes": d["n_nodes"],
                    })
        print(f"  {n} paires  ({(time.time()-t0)/n:.1f} s/paire)", flush=True)
        if max_pairs and n >= max_pairs:
            break
    return rows


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description="ondelettes + z-score intra-prompt")
    p.add_argument("cache_dir", nargs="?")
    p.add_argument("--selftest", action="store_true")
    p.add_argument("--layer-stride", type=int, default=8)
    p.add_argument("--head-stride", type=int, default=7)
    p.add_argument("--per-head", action="store_true")
    p.add_argument("--scales", default=",".join(str(s) for s in SCALES))
    p.add_argument("--widths", default=",".join(str(w) for w in WIDTHS))
    p.add_argument("--reduction", default="kron",
                   choices=["none", "drop", "renorm", "kron"])
    p.add_argument("--sparse", default="q90")
    p.add_argument("--kernel", default="mexican_hat", choices=list(KERNELS))
    p.add_argument("--surrogates", type=int, default=0,
                   help="calibration par prompt (couteux : 0 pour l'ignorer)")
    p.add_argument("--top-k", type=int, default=5)
    p.add_argument("--tail", default="both",
                   choices=["upper", "lower", "both"],
                   help="queue cherchee : la mesure dit que l'injection est "
                        "un creux d'energie, pas un pic")
    p.add_argument("--calibrated-rule", action="store_true",
                   help="regle argmin(p) par couple. Mesuree comme inferieure "
                        "a argmax|z| (pooled 0,508 contre 0,570) : le biais "
                        "vers les petites fenetres est la signature, pas un "
                        "defaut. Desactivee par defaut.")
    p.add_argument("--calibrate-scales", action="store_true",
                   help="remplace --scales par des echelles dont l'etalement "
                        "mesure couvre 3 a 30 tokens")
    p.add_argument("--max-pairs", type=int, default=None)
    p.add_argument("--csv", default=None)
    args = p.parse_args(argv)

    if args.selftest:
        return selftest()
    if not args.cache_dir:
        print("il faut un cache, ou --selftest", file=sys.stderr)
        return 2

    scales = tuple(float(x) for x in args.scales.split(","))
    widths = tuple(int(x) for x in args.widths.split(","))
    rows = run(args.cache_dir, args.layer_stride, args.head_stride,
               args.per_head, scales, args.reduction, args.sparse,
               args.kernel, args.surrogates, args.top_k, args.max_pairs,
               args.tail, args.calibrate_scales, widths,
               args.calibrated_rule)
    if not rows:
        print("aucune mesure", file=sys.stderr)
        return 1

    from .hypotheses_report import auroc

    print(f"\n{len(rows)} mesures · echelles {list(map(int, scales))} · "
          f"retrait {args.reduction} · {args.sparse} · noyau {args.kernel}\n")
    cells = sorted({(r["layer"], r["head"]) for r in rows})
    print("ANCIENNE regle argmax|z| contre NOUVELLE regle argmin(p), "
          "memes graphes, memes paires")
    print(f"{'couche':>7}{'tete':>6}{'n':>5}"
          f"{'pooled brut':>13}{'pooled cal.':>13}"
          f"{'app. brut':>11}{'app. cal.':>11}"
          f"{'IoU brut':>10}{'IoU cal.':>10}{'top-k':>8}")
    print("-" * 94)
    best = (0.0, None)
    y2 = None
    for l, h in cells:
        sub = [r for r in rows if r["layer"] == l and r["head"] == h]
        inj = {r["pair"]: r for r in sub if r["condition"] == "injected"}
        ctl = {r["pair"]: r for r in sub if r["condition"] == "control"}
        keys = sorted(set(inj) & set(ctl))
        if len(keys) < 5:
            continue
        y2 = np.r_[np.ones(len(keys)), np.zeros(len(keys))]

        def pair_auroc(field):
            a = np.array([inj[k][field] for k in keys], dtype=float)
            b = np.array([ctl[k][field] for k in keys], dtype=float)
            if not (np.isfinite(a).all() and np.isfinite(b).all()):
                return float("nan"), float("nan")
            return auroc(y2, np.r_[a, b]), float((a > b).mean())

        p_raw, q_raw = pair_auroc("score_raw")
        p_cal, q_cal = pair_auroc("score_zstd")
        iou_c = float(np.nanmean([inj[k]["iou"] for k in keys]))
        iou_r = float(np.nanmean([inj[k]["iou_raw"] for k in keys]))
        tk = float(np.mean([inj[k]["hit_topk"] for k in keys]))
        print(f"{l:>7}{h:>6}{len(keys):>5}{p_raw:>13.3f}{p_cal:>13.3f}"
              f"{q_raw:>11.3f}{q_cal:>11.3f}{iou_r:>10.3f}{iou_c:>10.3f}"
              f"{tk:>8.3f}")
        if np.isfinite(p_cal) and max(p_cal, 1 - p_cal) > best[0]:
            best = (max(p_cal, 1 - p_cal), (l, h, p_cal, q_cal))

    print()
    if best[1]:
        l, h, ap, aq = best[1]
        print(f"Meilleure cellule, regle calibree : couche {l}, tete {h} -- "
              f"pooled {ap:.3f}, apparie {aq:.3f}")
    nt = [r["n_tied"] for r in rows if np.isfinite(r.get("p_cal", np.nan))]
    if nt:
        print(f"Couples ex aequo au plancher de p : mediane {np.median(nt):.0f} "
              f"sur {len(scales)*len(widths)} -- d'ou le departage par z_std, "
              f"qui sert aussi de score.")
    print("pooled  = ce que voit un detecteur (pas de paire).")
    print("apparie = part des paires ou l'injecte depasse son controle.")
    print("brut    = argmax|z| a travers les couples (biaise vers w petit).")
    print("cal.    = argmin(p) par couple, score = z_std du vainqueur.")
    print("IoU     = recouvrement de la fenetre predite avec le vrai span.")
    sc = {}
    for r in rows:
        if r["condition"] == "injected" and np.isfinite(r["scale"]):
            key = (r["scale"], r.get("width", 0))
            sc[key] = sc.get(key, 0) + 1
    tot = max(sum(sc.values()), 1)
    print("\nCouple (echelle, largeur) retenu par la regle aveugle :")
    for k in sorted(sc, key=lambda x: -sc[x])[:8]:
        print(f"  s = {k[0]:>5.2f}  w = {k[1]:>3} : {100*sc[k]/tot:>5.1f} %")
    print("Une concentration sur la plus petite largeur est le biais "
          "d'agregation signale dans blind_decision, pas un resultat.")

    if args.csv:
        import csv as _csv
        os.makedirs(os.path.dirname(args.csv) or ".", exist_ok=True)
        fields = [k for k in rows[0] if k != "top"]
        with open(args.csv, "w", newline="", encoding="utf-8") as fh:
            w = _csv.DictWriter(fh, fieldnames=fields, extrasaction="ignore")
            w.writeheader()
            w.writerows(rows)
        print(f"\n-> {args.csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
