"""H6 : dynamique des proprietes structurelles et spectrales entre couches.

Pour une metrique ``M`` suivie sur les couches d'une meme tete :

    Delta M^(l) = M^(l+1) - M^(l)

Les statistiques de trajectoire sont **pre-declarees** et calculees sur
toutes les couches. Aucune couche n'est selectionnee sur son pouvoir
discriminant, et aucune n'est designee a priori comme interessante : le
protocole interdit explicitement de presupposer que les modifications
apparaissent dans les couches intermediaires.

Les quantites rendues :

    total_variation   somme des |Delta|, l'amplitude cumulee du parcours
    mean_abs_delta    variation moyenne entre couches consecutives
    max_abs_delta     plus grand saut, et sa profondeur relative
    n_sign_changes    nombre de changements de tendance, normalise
    drift             M(derniere) - M(premiere), le deplacement net
    roughness         ecart-type des Delta, la regularite du parcours
    depth_of_max      profondeur RELATIVE l/(L-1) du plus grand saut

La profondeur relative permet de comparer des architectures de profondeurs
differentes ; la profondeur absolue est conservee a cote, comme demande.
"""

from typing import Dict, Optional, Sequence

import numpy as np

EPS = 1e-12


_STATS = ("total_variation", "mean_abs_delta", "max_abs_delta",
          "depth_of_max", "depth_of_max_abs", "n_sign_changes", "drift",
          "roughness", "mean_level", "std_level", "n_valid_deltas",
          "n_dropped_deltas")


def trajectory_stats(traj: np.ndarray) -> Dict[str, float]:
    """Statistiques pre-declarees d'une trajectoire ``[L]`` sur les couches.

    **Les valeurs non finies ne sont jamais compactees.** Une version
    anterieure faisait ``m = m[isfinite(m)]`` puis ``np.diff`` : si la
    couche 5 etait manquante, la difference etait calculee entre les
    couches 4 et 6 et fabriquait une transition entre couches **non
    consecutives**. Le defaut gonflait la variation totale et pouvait
    inventer un changement de regime la ou il n'y avait qu'un trou.

    Ici les differences sont calculees sur le tableau complet, ce qui
    propage les NaN, puis seules les differences finies -- donc issues de
    deux couches reellement adjacentes -- sont conservees. Le nombre de
    differences retenues et le nombre ecartees sont rendus, de sorte qu'un
    trou soit visible dans les sorties au lieu d'etre absorbe.

    ``drift`` est calcule entre la premiere et la derniere couche finies, et
    vaut NaN si l'une des deux manque : l'extrapoler serait une invention.
    """
    m = np.asarray(traj, dtype=np.float64)
    L = m.size
    nan = {k: float("nan") for k in _STATS}
    if L < 3:
        return nan

    d_all = np.diff(m)                      # NaN la ou une couche manque
    good = np.isfinite(d_all)
    n_valid, n_dropped = int(good.sum()), int((~good).sum())
    if n_valid < 2:
        nan["n_valid_deltas"] = float(n_valid)
        nan["n_dropped_deltas"] = float(n_dropped)
        return nan

    d = d_all[good]
    pos = np.flatnonzero(good)              # profondeurs reelles conservees
    nz = d[np.abs(d) > EPS]
    sign_changes = int((np.diff(np.sign(nz)) != 0).sum()) if nz.size > 1 else 0
    j = int(pos[int(np.argmax(np.abs(d)))])

    fin = np.flatnonzero(np.isfinite(m))
    drift = (float(m[fin[-1]] - m[fin[0]]) if fin.size >= 2 else float("nan"))
    lvl = m[np.isfinite(m)]
    return {
        "total_variation": float(np.abs(d).sum()),
        "mean_abs_delta": float(np.abs(d).mean()),
        "max_abs_delta": float(np.abs(d).max()),
        "depth_of_max": float(j / max(L - 2, 1)),      # relative, dans [0,1]
        "depth_of_max_abs": float(j),                   # absolue
        "n_sign_changes": float(sign_changes / max(nz.size - 1, 1)),
        "drift": drift,
        "roughness": float(d.std()),
        "mean_level": float(lvl.mean()),
        "std_level": float(lvl.std()),
        "n_valid_deltas": float(n_valid),
        "n_dropped_deltas": float(n_dropped),
    }


def h6_trajectories(V: np.ndarray, metric_names: Sequence[str],
                    metrics: Optional[Sequence[str]] = None
                    ) -> Dict[str, np.ndarray]:
    """``V`` ``[couches, tetes, metriques]`` -> dict de tableaux ``[tetes]``.

    Une trajectoire par (tete, metrique) ; les statistiques sont calculees
    tete par tete, jamais sur des couches moyennees entre tetes.
    """
    V = np.asarray(V, dtype=np.float64)
    if V.ndim != 3:
        raise ValueError(f"attendu [L, H, M], recu {V.shape}")
    names = list(metric_names)
    chosen = list(metrics) if metrics is not None else names
    out: Dict[str, np.ndarray] = {}
    for name in chosen:
        if name not in names:
            continue
        k = names.index(name)
        per_head = [trajectory_stats(V[:, h, k]) for h in range(V.shape[1])]
        for stat in per_head[0]:
            out[f"traj_{name}__{stat}"] = np.array(
                [p[stat] for p in per_head], dtype=np.float64)
    return out


def partition_stability(parts_by_layer: Sequence[Dict[int, np.ndarray]],
                        k: int, ari_fn) -> Dict[str, float]:
    """Stabilite des partitions entre couches consecutives, a K fixe.

    ``parts_by_layer[l][k]`` est l'etiquetage de la couche ``l``. On mesure
    l'indice de Rand ajuste entre couches successives : eleve = la structure
    communautaire persiste en profondeur, bas = elle se reorganise.
    """
    vals = []
    for a, b in zip(parts_by_layer, parts_by_layer[1:]):
        if k in a and k in b:
            vals.append(ari_fn(a[k], b[k]))
    if not vals:
        return {"ari_mean": float("nan"), "ari_min": float("nan"),
                "ari_std": float("nan"), "ari_drift": float("nan")}
    v = np.asarray(vals, dtype=np.float64)
    return {"ari_mean": float(v.mean()), "ari_min": float(v.min()),
            "ari_std": float(v.std()),
            "ari_drift": float(v[-1] - v[0]) if v.size > 1 else 0.0}
