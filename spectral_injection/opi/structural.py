"""H1 a H4 : connectivite, frontieres, flux orientes, spectre de l'indicateur.

Aucun noeud n'est retire. Aucune reduction. Les puits d'attention font partie
du graphe etudie.

Conventions declarees, valables pour tout le module :

* ``A`` est la matrice d'attention ORIENTEE brute, ``A[i, j]`` = attention de
  la requete ``i`` vers la cle ``j``. Causale, donc triangulaire inferieure,
  lignes sommant a 1.
* ``W = (A + A^T) / 2``, diagonale mise a zero. Le retrait des boucles n'est
  pas un retrait de noeud : c'est la convention standard d'un laplacien de
  graphe, et sans elle une boucle ``W_ii`` modifie le spectre de ``L_sym``
  sans correspondre a aucune arete entre deux tokens. Choix declare.
* ``D_ii = sum_j W_ij``, ``L_sym = I - D^(-1/2) W D^(-1/2)``. Un noeud de
  degre nul recoit ``D^(-1/2)_ii = 0``, donc une ligne nulle hors diagonale
  et un 1 sur la diagonale : il devient une composante isolee, ce qui est le
  comportement correct.
* H3 travaille sur ``A`` sans symetrisation. H1, H2 et H4 sur ``W``.
"""

from typing import Dict, Optional, Sequence, Tuple

import numpy as np

EPS = 1e-12
#: Bandes spectrales pre-declarees, sur [0, 2] qui borne le spectre de L_sym.
BANDS: Tuple[Tuple[float, float], ...] = ((0.0, 0.5), (0.5, 1.0),
                                          (1.0, 1.5), (1.5, 2.0 + 1e-9))
BAND_NAMES = ("band_0_05", "band_05_1", "band_1_15", "band_15_2")
#: Tolerance de nullite d'une valeur propre, relative a la taille du graphe.
ZERO_TOL = 1e-8


# --------------------------------------------------------------------------
# Graphe
# --------------------------------------------------------------------------
def symmetrize(A: np.ndarray, remove_self_loops: bool = True) -> np.ndarray:
    """``A`` [T,T] ou [H,T,T] -> ``W`` symetrique."""
    A = np.asarray(A, dtype=np.float64)
    W = 0.5 * (A + np.swapaxes(A, -1, -2))
    if remove_self_loops:
        idx = np.arange(W.shape[-1])
        W[..., idx, idx] = 0.0
    return W


def degrees(W: np.ndarray) -> np.ndarray:
    return W.sum(-1)


def laplacian_sym(W: np.ndarray) -> np.ndarray:
    """``I - D^-1/2 W D^-1/2``, symetrise contre la derive numerique."""
    d = degrees(W)
    inv = np.where(d > EPS, 1.0 / np.sqrt(np.maximum(d, EPS)), 0.0)
    L = -W * inv[..., :, None] * inv[..., None, :]
    idx = np.arange(W.shape[-1])
    L[..., idx, idx] += 1.0
    return 0.5 * (L + np.swapaxes(L, -1, -2))


def spectrum(L: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Valeurs propres croissantes clippees a [0, 2], et vecteurs propres."""
    vals, vecs = np.linalg.eigh(L)
    return np.clip(vals, 0.0, 2.0), vecs


# --------------------------------------------------------------------------
# H1 — connectivite spectrale globale
# --------------------------------------------------------------------------
def h1_connectivity(vals: np.ndarray, zero_tol: float = ZERO_TOL
                    ) -> Dict[str, float]:
    """Resume du spectre. Aucune direction n'est presupposee.

    ``lambda_2`` est rendu tel quel, et ``lambda_first_nonzero`` est la
    premiere valeur propre au-dessus de la tolerance : les deux coincident
    sur un graphe connexe et divergent sinon. ``n_zero`` donne la
    multiplicite de la valeur propre nulle, c'est-a-dire le nombre de
    composantes connexes.
    """
    v = np.asarray(vals, dtype=np.float64)
    n = v.size
    nz = int((v <= zero_tol).sum())
    nonzero = v[v > zero_tol]
    return {
        "n_nodes": float(n),
        "lambda_2": float(v[1]) if n > 1 else float("nan"),
        "lambda_3": float(v[2]) if n > 2 else float("nan"),
        "lambda_max": float(v[-1]),
        "n_zero": float(nz),
        "lambda_first_nonzero": float(nonzero[0]) if nonzero.size else float("nan"),
        "eigengap_23": float(v[2] - v[1]) if n > 2 else float("nan"),
        "spectral_mean": float(v.mean()),
        "spectral_std": float(v.std()),
        "spectral_q25": float(np.quantile(v, 0.25)),
        "spectral_q50": float(np.quantile(v, 0.50)),
        "spectral_q75": float(np.quantile(v, 0.75)),
        "trace_over_n": float(v.sum() / max(n, 1)),
    }


# --------------------------------------------------------------------------
# H2 — frontieres structurelles de E
# --------------------------------------------------------------------------
def h2_boundary(W: np.ndarray, mask_E: np.ndarray) -> Dict[str, float]:
    """Coupe, conductance, Ncut, volumes. Cas degeneres rendus en NaN."""
    W = np.asarray(W, dtype=np.float64)
    e = np.asarray(mask_E, dtype=bool)
    if e.all() or (~e).all():
        return {k: float("nan") for k in
                ("cut", "vol_E", "vol_Ebar", "vol_total", "conductance",
                 "ncut", "cut_over_vol_E", "size_E", "size_Ebar")}
    d = degrees(W)
    cut = float(W[np.ix_(e, ~e)].sum())
    vol_e = float(d[e].sum())
    vol_b = float(d[~e].sum())
    return {
        "cut": cut,
        "vol_E": vol_e,
        "vol_Ebar": vol_b,
        "vol_total": float(d.sum()),
        "conductance": cut / max(min(vol_e, vol_b), EPS),
        "ncut": cut * (1.0 / max(vol_e, EPS) + 1.0 / max(vol_b, EPS)),
        "cut_over_vol_E": cut / max(vol_e, EPS),
        "size_E": float(e.sum()),
        "size_Ebar": float((~e).sum()),
    }


def dirichlet_combinatorial(W: np.ndarray, x: np.ndarray) -> float:
    """``x^T (D - W) x``. Pour un indicateur binaire, vaut exactement la
    coupe -- c'est le controle mathematique de ``h2_boundary``."""
    W = np.asarray(W, dtype=np.float64)
    x = np.asarray(x, dtype=np.float64)
    return float(x @ (np.diag(degrees(W)) - W) @ x)


# --------------------------------------------------------------------------
# H3 — flux orientes entre regions
# --------------------------------------------------------------------------
def h3_flows(A: np.ndarray, regions: np.ndarray) -> Dict[str, float]:
    """Flux sur l'attention ORIENTEE brute, sans symetrisation.

    ``A[i, j]`` = attention de la requete ``i`` vers la cle ``j``. Les sommes
    brutes et les versions normalisees par ``|source|`` sont rendues.

    ``F_{U->E}`` est calcule et doit valoir 0 : U precede E et le masque
    causal l'impose. Ce n'est pas une propriete de securite, c'est une
    contrainte d'architecture, et sa valeur sert de controle d'integrite.
    """
    A = np.asarray(A, dtype=np.float64)
    out = {}
    idx = {r: np.flatnonzero(regions == r) for r in ("U", "E", "T")}
    for src in ("E", "U", "T"):
        i = idx[src]
        if i.size == 0:
            continue
        for dst in ("U", "E", "T"):
            j = idx[dst]
            if j.size == 0:
                continue
            f = float(A[np.ix_(i, j)].sum())
            out[f"F_{src}_to_{dst}"] = f
            out[f"Fbar_{src}_to_{dst}"] = f / i.size
    for r in ("U", "E", "T"):
        out[f"size_{r}"] = float(idx[r].size)
    return out


# --------------------------------------------------------------------------
# H4 — distribution spectrale de l'indicateur de E
# --------------------------------------------------------------------------
def h4_spectral_signal(vals: np.ndarray, vecs: np.ndarray,
                       mask_E: np.ndarray, variant: str = "A",
                       zero_tol: float = ZERO_TOL) -> Dict[str, float]:
    """Repartition de l'energie de ``1_E`` sur les frequences du graphe.

    ``variant="A"`` : signal brut normalise ``x / ||x||``.
    ``variant="B"`` : signal projete hors du noyau spectral,
    ``x_perp = (I - U_0 U_0^T) x`` puis normalise. ``U_0`` est la base des
    vecteurs propres de valeur propre nulle -- pour ``L_sym`` connexe, c'est
    ``D^{1/2} 1`` normalise, et surtout PAS le vecteur constant : soustraire
    la moyenne arithmetique serait faux.

    Les quantites rendues sont invariantes par changement de base a
    l'interieur d'un sous-espace propre : les masses de bande somment des
    ``p_k`` sur des valeurs propres egales, et le centroide est un quotient
    de Rayleigh. Aucune ne depend du choix des vecteurs propres en cas de
    multiplicite.
    """
    vals = np.asarray(vals, dtype=np.float64)
    vecs = np.asarray(vecs, dtype=np.float64)
    x = np.asarray(mask_E, dtype=np.float64)

    if variant == "B":
        null = vals <= zero_tol
        if null.any():
            U0 = vecs[:, null]
            x = x - U0 @ (U0.T @ x)
    nrm = np.linalg.norm(x)
    if nrm <= EPS:
        return {k: float("nan") for k in
                (*BAND_NAMES, "hfer", "centroid", "variance", "ipr_spectral",
                 "rayleigh", "signal_norm")}
    xh = x / nrm

    alpha = vecs.T @ xh
    p = alpha ** 2
    p = p / max(p.sum(), EPS)

    out = {f"{name}": float(p[(vals >= lo) & (vals < hi)].sum())
           for name, (lo, hi) in zip(BAND_NAMES, BANDS)}
    out["hfer"] = float(p[vals >= 1.0].sum())
    centroid = float((p * vals).sum())
    out["centroid"] = centroid
    out["variance"] = float((p * (vals - centroid) ** 2).sum())
    out["ipr_spectral"] = float((p ** 2).sum())
    # quotient de Rayleigh : doit egaler le centroide au niveau machine
    L_x = vecs @ (vals * (vecs.T @ xh))
    out["rayleigh"] = float(xh @ L_x)
    out["signal_norm"] = float(nrm)
    return out


# --------------------------------------------------------------------------
# Assemblage par couche / tete
# --------------------------------------------------------------------------
def analyse_layer(A_layer: np.ndarray, regions: np.ndarray,
                  per_head: bool = True, with_communities: bool = False,
                  seed: int = 0):
    """Une couche ``[H, T, T]`` -> dict de tableaux ``[H]`` (ou ``[1]``).

    **Deux variantes de graphe sont calculees**, decision methodologique
    validee : la boucle ``W_ii`` est conservee dans la variante principale et
    retiree dans la variante de comparaison. Aucun noeud n'est supprime dans
    l'une ni dans l'autre -- seule l'auto-arete change.

        prefixe (aucun)   W avec diagonale        <- principale
        prefixe ``nsl_``  W a diagonale nulle     <- variante de controle

    H3 ne depend pas de ``W`` : il travaille sur ``A`` orientee brute, dont
    la diagonale (l'auto-attention) est conservee dans tous les cas. Il n'est
    donc calcule qu'une fois.

    ``per_head=False`` moyenne les tetes AVANT la construction du graphe et
    le signale par une seule ligne : c'est une agregation declaree, pas un
    resume d'agregations.
    """
    A = np.asarray(A_layer, dtype=np.float64)
    if A.ndim != 3:
        raise ValueError(f"attendu [H, T, T], recu {A.shape}")
    heads = range(A.shape[0]) if per_head else [None]
    mask_E = regions == "E"
    xE = mask_E.astype(float)
    rows, parts = [], []
    for h in heads:
        Ah = A[h] if h is not None else A.mean(0)
        rec = {f"h3_{k}": v for k, v in h3_flows(Ah, regions).items()}
        head_parts = None
        for prefix, rm_loops in (("", False), ("nsl_", True)):
            W = symmetrize(Ah, remove_self_loops=rm_loops)
            vals, vecs = spectrum(laplacian_sym(W))
            rec.update({f"{prefix}h1_{k}": v
                        for k, v in h1_connectivity(vals).items()})
            rec.update({f"{prefix}h2_{k}": v
                        for k, v in h2_boundary(W, mask_E).items()})
            for var in ("A", "B"):
                rec.update({f"{prefix}h4{var}_{k}": v for k, v in
                            h4_spectral_signal(vals, vecs, mask_E, var).items()})
            rec[f"{prefix}h2_dirichlet_comb"] = dirichlet_combinatorial(W, xE)
            # H5 n'est calcule que sur la variante principale (avec boucles)
            if with_communities and prefix == "":
                from .communities import h5_communities
                h5, head_parts = h5_communities(
                    vals, vecs, degrees(W), regions, seed=seed)
                rec.update({f"h5_{k}": v for k, v in h5.items()})
        rows.append(rec)
        parts.append(head_parts)
    keys = rows[0].keys()
    met = {k: np.array([r[k] for r in rows], dtype=np.float64) for k in keys}
    return (met, parts) if with_communities else (met, None)
