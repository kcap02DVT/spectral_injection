"""Metriques par tete via ``spectral-trust`` ; quantites de region en local.

Ce que la bibliotheque fait, elle le fait seule : ``per_head_metrics`` rend
pour chaque tete d'une couche ``fiedler_value`` (lambda_2),
``connectivity_ratio``, ``spectral_entropy_norm``, ``hfer`` (masse propre de
la moitie superieure du spectre, par RANG) et ``spectral_radius``. Elle
utilise ``torch.linalg.eigvalsh`` dense et batche -- la voie ARPACK tronquee
de ``SpectralAnalyzer`` n'est PAS empruntee ici.

Ce qu'elle ne rend pas : les vecteurs propres. Le vecteur de Fiedler est
donc obtenu en construisant le laplacien **par la bibliotheque**
(``GraphConstructor``) puis en le diagonalisant ; lambda_2 ainsi obtenu est
compare a ``fiedler_value`` comme controle d'integrite.

Quantites de region (coupe, volumes, Ncut, flux) : locales, sur le graphe
COMPLET avec des masques. Jamais ``token_span``.
"""

from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

EPS = 1e-12


def library_config():
    """Convention declaree. ``remove_self_loops=False`` et ``sym`` sont les
    defauts de la bibliotheque ; ``eigen_solver='dense'`` ne concerne pas
    ``per_head_metrics`` (toujours dense) mais est fixe pour coherence si
    ``SpectralAnalyzer`` etait appele."""
    from spectral_trust import GSPConfig
    return GSPConfig(symmetrization="symmetric", normalization="sym",
                     remove_self_loops=False, eigen_solver="dense",
                     verbose=False, save_plots=False, display_plots=False,
                     save_intermediate=False)


def per_head_long(att: Sequence[np.ndarray], cfg, example_id, condition: str
                  ) -> Tuple[pd.DataFrame, List[Dict]]:
    """Toutes les couches, TOUTES les tetes -> tableau long.

    Colonnes : exemple_id, condition, couche, tete, metrique, valeur.
    Un echec sur une tete est journalise et la tete rendue NaN, jamais
    remplacee par une moyenne.
    """
    import torch
    from spectral_trust import per_head_metrics
    rows, errors = [], []
    for l, A in enumerate(att):
        try:
            d = per_head_metrics(torch.from_numpy(np.ascontiguousarray(A)),
                                 config=cfg, layer=l)
            vals, names = d.values, d.metric_names
            for h in range(vals.shape[0]):
                for m, v in zip(names, vals[h]):
                    rows.append((example_id, condition, l, h, m, float(v)))
        except Exception as e:                              # noqa: BLE001
            errors.append({"exemple_id": example_id, "condition": condition,
                           "couche": l, "erreur": f"{type(e).__name__}: {e}"})
            for h in range(A.shape[0]):
                for m in ("fiedler_value", "connectivity_ratio",
                          "spectral_entropy_norm", "hfer", "spectral_radius"):
                    rows.append((example_id, condition, l, h, m, float("nan")))
    df = pd.DataFrame(rows, columns=["exemple_id", "condition", "couche",
                                     "tete", "metrique", "valeur"])
    return df, errors


def laplacian_library(A_head: np.ndarray, cfg):
    """W et L_sym d'UNE tete, construits par ``spectral-trust``."""
    import torch
    from spectral_trust import GraphConstructor
    gc = GraphConstructor(cfg)
    a = torch.from_numpy(np.ascontiguousarray(A_head, dtype=np.float32))
    W = gc.symmetrize_attention(a[None, None])[0, 0]          # [T, T]
    L = gc.construct_laplacian(W[None])[0]                    # [T, T]
    L = 0.5 * (L + L.T)
    return W.numpy().astype(np.float64), L.numpy().astype(np.float64)


def fiedler(A_head: np.ndarray, cfg) -> Dict:
    """Spectre complet et vecteur de Fiedler d'une tete, sur le laplacien
    de la bibliotheque. Le signe de u_2 est fixe (premiere composante non
    nulle positive) pour la reproductibilite des traces."""
    W, L = laplacian_library(A_head, cfg)
    vals, vecs = np.linalg.eigh(L)
    vals = np.clip(vals, 0.0, 2.0)
    u2 = vecs[:, 1].copy()
    nz = np.flatnonzero(np.abs(u2) > 1e-10)
    if nz.size and u2[nz[0]] < 0:
        u2 = -u2
    return {"lambda_2": float(vals[1]), "lambda_max": float(vals[-1]),
            "eigenvalues": vals, "fiedler_vector": u2, "W": W}


def region_quantities(A_head: np.ndarray, regions: np.ndarray, cfg,
                      region: str = "U") -> Dict[str, float]:
    """Coupe, volumes, conductance, Ncut de ``region`` contre son
    complement, sur ``W`` de la bibliotheque ; flux orientes sur ``A`` brute.
    Graphe complet, masques -- aucune extraction de sous-graphe."""
    W, _ = laplacian_library(A_head, cfg)
    m = regions == region
    if m.sum() == 0 or (~m).sum() == 0:
        return {k: float("nan") for k in ("cut", "vol_R", "vol_Rbar",
                                           "conductance", "ncut")}
    cut = float(W[np.ix_(m, ~m)].sum())
    d = W.sum(1)
    vr, vb = float(d[m].sum()), float(d[~m].sum())
    out = {"cut": cut, "vol_R": vr, "vol_Rbar": vb,
           "conductance": cut / max(min(vr, vb), EPS),
           "ncut": cut * (1.0 / max(vr, EPS) + 1.0 / max(vb, EPS))}
    A = np.asarray(A_head, dtype=np.float64)
    idx = {r: np.flatnonzero(regions == r) for r in ("S", "U", "T")}
    for s in ("S", "U", "T"):
        for t in ("S", "U", "T"):
            if idx[s].size and idx[t].size:
                out[f"Fbar_{s}_to_{t}"] = float(A[np.ix_(idx[s], idx[t])].sum()) / idx[s].size
    return out
