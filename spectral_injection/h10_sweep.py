"""H10 : le verdict depend-il de la facon de construire le graphe ?

Lit le cache d'attention, donc ne recharge jamais le modele. Rejoue le test
de clustering spectral (serie B) sous toutes les combinaisons de :

    plongement   hamilton    K-1 colonnes, lignes brutes      (ce qui a ete
                             teste, et qui suit la lettre du 2.3.3)
                 vluxburg    K colonnes, vecteur trivial inclus
                 njw         K colonnes, lignes ramenees a la norme 1
                 shimalik    D^-1/2 U, K colonnes

    graphe       dense       aucune sparsification
                 q80/q90/q95 seuil au quantile des poids hors diagonale
                 topk16      les 16 plus fortes aretes par noeud

Le k-means passe en k-means++ avec relances, et les clusters vides sont
reaffectes : trois defauts de l'implementation precedente qui pouvaient a
eux seuls produire un resultat negatif.

Pour chaque combinaison on rapporte la meme chose qu'en serie B -- F1 du
meilleur cluster et part des paires ou un cluster est majoritairement
injecte -- dans les deux conditions. Le verdict de la serie B ne tient que
si l'ecart injecte-controle reste nul partout.
"""

import argparse
import os
import sys

import numpy as np

from .attn_cache import iter_cache
from .prompts import ROLE_INJECTION, ROLE_TEMPLATE

EMB = ("hamilton", "vluxburg", "njw", "shimalik")
GRAPH = ("dense", "q80", "q90", "q95", "topk16")
K_VALUES = (2, 5, 12)
EPS = 1e-12


def _graph(A_layer, roles, mode):
    """Attention [H,T,T] -> (W filtre, roles filtres). Symetrise, sans puits."""
    A = np.asarray(A_layer, dtype=np.float64).mean(0)
    W = 0.5 * (A + A.T)
    np.fill_diagonal(W, 0.0)
    keep = roles != ROLE_TEMPLATE
    keep[0] = False
    sink = int(W.sum(0).argmax())
    keep[sink] = False
    W = W[np.ix_(keep, keep)]
    if mode.startswith("q"):
        off = W[W > 0]
        if off.size:
            W = np.where(W >= np.quantile(off, int(mode[1:]) / 100.0), W, 0.0)
    elif mode.startswith("topk"):
        k = int(mode[4:])
        if W.shape[0] > k:
            thr = np.partition(W, -k, axis=1)[:, -k][:, None]
            M = W >= thr
            W = np.where(M | M.T, W, 0.0)
    return W, roles[keep]


def _embed(W, kind, k):
    """Plongement spectral selon la recette demandee."""
    d = W.sum(1)
    dinv = 1.0 / np.sqrt(np.maximum(d, EPS))
    L = np.eye(W.shape[0]) - (W * dinv[:, None]) * dinv[None, :]
    vals, vecs = np.linalg.eigh((L + L.T) / 2.0)
    if kind == "hamilton":
        U = vecs[:, 1:max(k, 2)]
    else:
        U = vecs[:, :k]
        if kind == "njw":
            n = np.linalg.norm(U, axis=1, keepdims=True)
            U = U / np.maximum(n, EPS)
        elif kind == "shimalik":
            U = U * dinv[:, None]
    return U


def _kmeanspp(X, k, rng, restarts=5, iters=60):
    """k-means++ avec relances, clusters vides reinitialises."""
    best, best_cost = None, np.inf
    n = X.shape[0]
    k = min(k, n)
    for _ in range(restarts):
        c = [X[rng.integers(n)]]
        for _ in range(k - 1):
            d = np.min(((X[:, None, :] - np.array(c)[None]) ** 2).sum(-1), 1)
            p = d / max(d.sum(), EPS)
            c.append(X[rng.choice(n, p=p)])
        C = np.array(c)
        lab = np.zeros(n, int)
        for _ in range(iters):
            dist = ((X[:, None, :] - C[None]) ** 2).sum(-1)
            new = dist.argmin(1)
            if np.array_equal(new, lab):
                break
            lab = new
            for j in range(k):
                pts = X[lab == j]
                C[j] = pts.mean(0) if len(pts) else X[rng.integers(n)]
        cost = float(((X - C[lab]) ** 2).sum())
        if cost < best_cost:
            best, best_cost = lab.copy(), cost
    return best


def _score(lab, truth):
    """F1 du meilleur cluster, et 1 si un cluster est majoritairement injecte."""
    n_inj = int(truth.sum())
    if n_inj == 0:
        return 0.0, 0.0
    f1, maj = 0.0, 0.0
    for c in np.unique(lab):
        m = lab == c
        inter = float((m & truth).sum())
        prec, rec = inter / max(m.sum(), 1), inter / n_inj
        if prec > 0.5:
            maj = 1.0
        if prec + rec:
            f1 = max(f1, 2 * prec * rec / (prec + rec))
    return f1, maj


def main(argv=None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("cache_dir")
    p.add_argument("--layer-stride", type=int, default=4)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args(argv)

    rng = np.random.default_rng(args.seed)
    acc = {}
    npairs = 0
    for att_i, att_c, roles_i, roles_c, meta in iter_cache(args.cache_dir):
        npairs += 1
        layers = range(0, att_i.shape[0], args.layer_stride)
        for cond, (att, roles) in enumerate(((att_i, roles_i), (att_c, roles_c))):
            for l in layers:
                for g in GRAPH:
                    W, r = _graph(att[l], roles, g)
                    truth = r == ROLE_INJECTION
                    if W.shape[0] < 20 or truth.sum() == 0:
                        continue
                    for e in EMB:
                        for k in K_VALUES:
                            U = _embed(W, e, k)
                            lab = _kmeanspp(U, k, rng)
                            f1, maj = _score(lab, truth)
                            key = (g, e, k)
                            a = acc.setdefault(key, np.zeros((2, 3)))
                            a[cond] += (f1, maj, 1)
        if npairs % 10 == 0:
            print(f"  {npairs} paires traitees", flush=True)

    print(f"\n{npairs} paires, couches tous les {args.layer_stride}\n")
    print(f"{'graphe':<8}{'plongement':<11}{'K':>3}"
          f"{'F1 inj':>9}{'F1 ctl':>9}{'ecart':>8}"
          f"{'maj inj':>9}{'maj ctl':>9}{'ecart':>8}")
    print("-" * 74)
    flagged = []
    for g in GRAPH:
        for e in EMB:
            for k in K_VALUES:
                a = acc.get((g, e, k))
                if a is None or a[0, 2] == 0:
                    continue
                fi, mi = a[0, 0] / a[0, 2], a[0, 1] / a[0, 2]
                fc, mc = a[1, 0] / a[1, 2], a[1, 1] / a[1, 2]
                print(f"{g:<8}{e:<11}{k:>3}{fi:>9.3f}{fc:>9.3f}{fi-fc:>+8.3f}"
                      f"{mi:>9.3f}{mc:>9.3f}{mi-mc:>+8.3f}")
                if fi - fc > 0.05:
                    flagged.append((g, e, k, fi - fc))
    print()
    if flagged:
        print("Combinaisons ou le clustering separe l'injection (ecart F1 > 0,05) :")
        for g, e, k, d in sorted(flagged, key=lambda x: -x[3]):
            print(f"   {g} / {e} / K={k} : ecart {d:+.3f}")
        print("Le verdict de la serie B serait alors un artefact de construction.")
    else:
        print("Aucune combinaison ne separe l'injection : le verdict de la "
              "serie B tient, il n'etait pas un artefact du graphe choisi.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
