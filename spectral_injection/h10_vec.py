"""Tache 3 : vague 8 (H10) vectorisee -- le verdict depend-il du graphe ?

Remplace la boucle de ``h10_sweep.py``, qui n'a jamais tourne et qui ne
pouvait pas tourner en temps raisonnable : elle appelait ``_embed`` -- donc
une decomposition propre complete -- a l'interieur de la boucle sur les
plongements et les K, soit 60 decompositions par graphe au lieu d'une.

Ici le spectre ne depend que du seuillage, jamais du plongement ni de K :

    pour chaque seuillage          une decomposition propre, par lot
      pour chaque plongement       une lecture des memes vecteurs propres
        pour chaque K              un k-means++ par lot

ce qui divise le cout par 20 a resultat identique. Le reste est vectorise
sur la dimension de lot (les tetes, ou les couches si l'attention est
moyennee) : un seul ``eigh`` empile, un seul k-means batche.

Ce qui est balaye :

    seuillage    dense        aucune sparsification
                 q80 q90 q95  seuil au quantile des poids hors diagonale
                 topk16       16 aretes les plus fortes par noeud, OR
    plongement   hamilton     u_2..u_K          (K-1 colonnes, lignes brutes)
                 vluxburg     u_1..u_K          (vecteur trivial inclus)
                 njw          idem, lignes ramenees a la norme 1
                 shimalik     D^-1/2 [u_1..u_K]
    K            2, 5, 12

Le laplacien est le symetrique normalise dans tous les cas, sans exception :
L = I - D^-1/2 W D^-1/2. C'est la contrainte qui rend les spectres
comparables entre prompts de longueurs differentes, et la lever ferait
varier deux choses a la fois.

``--keep-sinks`` ajoute la question de la tache 1 au meme balayage : si le
verdict negatif de la serie B n'est un artefact ni de la sparsification ni
du retrait des puits, il tient pour de bon.

Ce qui est rapporte, dans les deux conditions :

    f1        du cluster qui recouvre le mieux le span (etiquettes)
    majority  part des graphes ou un cluster est majoritairement insere
    contig    contiguite du cluster le plus contigu -- AVEUGLE
    hit       ce cluster choisi a l'aveugle est-il le span (f1 > 0.5)

Seul l'ecart injecte - controle compte : un f1 eleve dans les deux
conditions veut dire que le clustering localise des insertions, pas des
injections.

    python -m spectral_injection.h10_vec cache/attn/MODELE/SPLIT \\
        --layer-stride 4 --per-head
"""

import argparse
import os
import sys
import time
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from .prompts import ROLE_INJECTION, ROLE_TEMPLATE

EMB = ("hamilton", "vluxburg", "njw", "shimalik")
GRAPH = ("dense", "q80", "q90", "q95", "topk16")
K_VALUES = (2, 5, 12)
EPS = 1e-12


# --------------------------------------------------------------------------
# Graphes, par lot
# --------------------------------------------------------------------------
def build_batch(attn_layer: np.ndarray, roles: np.ndarray,
                per_head: bool = True, keep_sinks: bool = False
                ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """[H,T,T] -> (W [B,n,n], roles gardes [n], positions d'origine [n]).

    Le filtrage des noeuds est identique a celui de ``h10_sweep._graph`` pour
    que les deux soient comparables : template, token 0, et le puits
    (argmax de masse recue) sur le graphe moyenne. Le choix du puits est
    fait une fois sur la moyenne des tetes, pas tete par tete, sinon les
    lots n'auraient pas le meme support et ne seraient plus empilables.
    """
    A = np.asarray(attn_layer, dtype=np.float64)
    if A.ndim != 3:
        raise ValueError(f"attendu [H,T,T], recu {A.shape}")
    Ab = A if per_head else A.mean(0)[None]
    W = 0.5 * (Ab + np.transpose(Ab, (0, 2, 1)))
    idx = np.arange(W.shape[-1])
    W[:, idx, idx] = 0.0

    keep = roles != ROLE_TEMPLATE
    if not keep_sinks:
        mean_W = W.mean(0)
        keep[0] = False
        keep[int(mean_W.sum(0).argmax())] = False
    pos = np.flatnonzero(keep)
    return W[:, keep][:, :, keep], roles[keep], pos


def threshold(W: np.ndarray, mode: str) -> np.ndarray:
    """Sparsification, par lot. ``W`` est [B,n,n] symetrique a diagonale nulle."""
    if mode == "dense":
        return W
    if mode.startswith("q"):
        q = int(mode[1:]) / 100.0
        out = np.empty_like(W)
        for b in range(W.shape[0]):               # quantile par element du lot
            off = W[b][W[b] > 0]
            if off.size == 0:
                out[b] = W[b]
                continue
            out[b] = np.where(W[b] >= np.quantile(off, q), W[b], 0.0)
        return out
    if mode.startswith("topk"):
        k = int(mode[4:])
        n = W.shape[-1]
        if n <= k:
            return W
        thr = np.partition(W, -k, axis=-1)[..., -k][..., None]
        M = W >= thr
        M = M | np.transpose(M, (0, 2, 1))        # OR : le graphe reste symetrique
        return np.where(M, W, 0.0)
    raise ValueError(f"seuillage inconnu : {mode}")


def laplacian_batch(W: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """L = I - D^-1/2 W D^-1/2 par lot, et D^-1/2 (pour shimalik)."""
    d = W.sum(-1)
    dinv = np.where(d > EPS, 1.0 / np.sqrt(np.maximum(d, EPS)), 0.0)
    L = -W * dinv[:, :, None] * dinv[:, None, :]
    idx = np.arange(W.shape[-1])
    L[:, idx, idx] += 1.0
    return 0.5 * (L + np.transpose(L, (0, 2, 1))), dinv


def eigh_batch(L: np.ndarray, backend: str = "numpy") -> Tuple[np.ndarray, np.ndarray]:
    """Decomposition propre empilee : une seule fois par seuillage."""
    if backend == "torch":
        import torch
        t = torch.from_numpy(L)
        vals, vecs = torch.linalg.eigh(t)
        return vals.numpy(), vecs.numpy()
    return np.linalg.eigh(L)


def embed(vecs: np.ndarray, dinv: np.ndarray, kind: str, k: int) -> np.ndarray:
    """Plongement spectral, par lot : [B,n,n] -> [B,n,d]."""
    if kind == "hamilton":
        U = vecs[:, :, 1:max(k, 2)]
    else:
        U = vecs[:, :, :k]
        if kind == "njw":
            nrm = np.linalg.norm(U, axis=-1, keepdims=True)
            U = U / np.maximum(nrm, EPS)
        elif kind == "shimalik":
            U = U * dinv[:, :, None]
    return np.ascontiguousarray(U)


# --------------------------------------------------------------------------
# k-means++ batche
# --------------------------------------------------------------------------
def _sq_dist(X: np.ndarray, C: np.ndarray) -> np.ndarray:
    """[B,n,d] x [B,k,d] -> [B,n,k], par produit matriciel (BLAS)."""
    return np.maximum(
        (X ** 2).sum(-1)[:, :, None]
        - 2.0 * np.einsum("bnd,bkd->bnk", X, C)
        + (C ** 2).sum(-1)[:, None, :], 0.0)


def _pick(p: np.ndarray, rng) -> np.ndarray:
    """Un indice par element du lot, tire selon les poids p [B,n]."""
    s = p.sum(-1, keepdims=True)
    q = np.where(s > EPS, p / np.maximum(s, EPS), 1.0 / p.shape[-1])
    c = np.cumsum(q, axis=-1)
    u = rng.random((p.shape[0], 1))
    return (c < u).sum(-1).clip(0, p.shape[-1] - 1)


def kmeanspp_batch(X: np.ndarray, k: int, rng, restarts: int = 5,
                   iters: int = 60) -> np.ndarray:
    """k-means++ batche, relances independantes par element du lot : [B,n].

    Le meilleur essai est retenu *par element du lot* et non globalement :
    chaque tete est un probleme de clustering distinct, et garder le meilleur
    cout global melangerait des graphes differents. Les clusters vides sont
    reinitialises sur le point le plus eloigne de son centre -- sans ca, un
    cluster vide fait silencieusement tomber K a K-1 et biaise le balayage
    vers les petits K.
    """
    B, n, d = X.shape
    k = min(k, n)
    best_lab = np.zeros((B, n), dtype=np.int64)
    best_cost = np.full(B, np.inf)
    ar = np.arange(B)

    for _ in range(restarts):
        C = np.empty((B, k, d))
        C[:, 0] = X[ar, rng.integers(n, size=B)]
        for j in range(1, k):
            d2 = _sq_dist(X, C[:, :j]).min(-1)             # [B,n]
            C[:, j] = X[ar, _pick(d2, rng)]
        lab = np.full((B, n), -1, dtype=np.int64)
        for _ in range(iters):
            dist = _sq_dist(X, C)
            new = dist.argmin(-1)
            if np.array_equal(new, lab):
                break
            lab = new
            oh = np.zeros((B, n, k))
            oh[ar[:, None], np.arange(n)[None, :], lab] = 1.0
            cnt = oh.sum(1)                                 # [B,k]
            C = np.einsum("bnk,bnd->bkd", oh, X) / np.maximum(cnt, 1.0)[:, :, None]
            empty = cnt == 0
            if empty.any():
                far = _sq_dist(X, C).min(-1).argmax(-1)     # [B]
                bi, ki = np.nonzero(empty)
                C[bi, ki] = X[bi, far[bi]]
        cost = np.take_along_axis(_sq_dist(X, C), lab[:, :, None], -1).sum((1, 2))
        better = cost < best_cost
        best_lab[better] = lab[better]
        best_cost[better] = cost[better]
    return best_lab


# --------------------------------------------------------------------------
# Scores
# --------------------------------------------------------------------------
def score_batch(lab: np.ndarray, truth: np.ndarray, pos: np.ndarray,
                k: int) -> Dict[str, np.ndarray]:
    """f1 / majority (etiquettes) et contig / hit (aveugle), par lot."""
    B, n = lab.shape
    n_inj = float(truth.sum())
    oh = np.zeros((B, n, k))
    oh[np.arange(B)[:, None], np.arange(n)[None, :], lab] = 1.0
    size = oh.sum(1)                                        # [B,k]
    inter = np.einsum("bnk,n->bk", oh, truth.astype(float))
    prec = inter / np.maximum(size, 1.0)
    rec = inter / max(n_inj, 1.0)
    f1 = np.where(prec + rec > 0, 2 * prec * rec / np.maximum(prec + rec, EPS), 0.0)

    # Contiguite : plus longue suite de positions consecutives / taille.
    contig = np.zeros((B, k))
    for j in range(k):
        for b in range(B):
            m = lab[b] == j
            if m.sum() < 3:
                continue
            p = pos[m]
            run = cur = 1
            for a, z in zip(p, p[1:]):
                cur = cur + 1 if z == a + 1 else 1
                run = max(run, cur)
            contig[b, j] = run / m.sum()
    pick = contig.argmax(-1)
    rows = np.arange(B)
    return {
        "f1": f1.max(-1),
        "majority": (prec > 0.5).any(-1).astype(float),
        "contig": contig[rows, pick],
        "hit": (f1[rows, pick] > 0.5).astype(float),
        "f1_blind": f1[rows, pick],
    }


# --------------------------------------------------------------------------
# Balayage
# --------------------------------------------------------------------------
def sweep_layer(attn_layer: np.ndarray, roles: np.ndarray, rng,
                per_head: bool = True, keep_sinks: bool = False,
                graphs: Sequence[str] = GRAPH, embs: Sequence[str] = EMB,
                ks: Sequence[int] = K_VALUES, backend: str = "numpy",
                restarts: int = 5) -> Dict[Tuple[str, str, int], Dict[str, np.ndarray]]:
    """Toutes les combinaisons pour une couche. Une eigh par seuillage."""
    W, r, pos = build_batch(attn_layer, roles, per_head, keep_sinks)
    truth = r == ROLE_INJECTION
    out: Dict[Tuple[str, str, int], Dict[str, np.ndarray]] = {}
    if W.shape[-1] < 20 or truth.sum() == 0:
        return out
    for g in graphs:
        Wg = threshold(W, g)
        L, dinv = laplacian_batch(Wg)
        _, vecs = eigh_batch(L, backend)
        for e in embs:
            for k in ks:
                U = embed(vecs, dinv, e, k)
                lab = kmeanspp_batch(U, k, rng, restarts=restarts)
                out[(g, e, k)] = score_batch(lab, truth, pos, min(k, U.shape[1]))
    return out


def run(cache_dir: str, layer_stride: int = 4, per_head: bool = True,
        keep_sinks: bool = False, seed: int = 0, backend: str = "numpy",
        graphs: Sequence[str] = GRAPH, embs: Sequence[str] = EMB,
        ks: Sequence[int] = K_VALUES, restarts: int = 5,
        max_pairs: Optional[int] = None) -> Tuple[Dict, int, int]:
    """Accumule sur tout le cache. Rend (sommes, n_paires, n_graphes)."""
    from .attn_cache import iter_cache

    rng = np.random.default_rng(seed)
    acc: Dict[Tuple[str, str, int], np.ndarray] = {}
    n_pairs = n_graphs = 0
    t0 = time.time()
    for att_i, att_c, roles_i, roles_c, meta in iter_cache(cache_dir):
        n_pairs += 1
        for l in range(0, att_i.shape[0], layer_stride):
            for cond, att, roles in ((0, att_i, roles_i), (1, att_c, roles_c)):
                res = sweep_layer(att[l], roles, rng, per_head, keep_sinks,
                                  graphs, embs, ks, backend, restarts)
                for i, (key, sc) in enumerate(res.items()):
                    a = acc.setdefault(key, np.zeros((2, 6)))
                    a[cond] += (sc["f1"].sum(), sc["majority"].sum(),
                                sc["contig"].sum(), sc["hit"].sum(),
                                sc["f1_blind"].sum(), sc["f1"].size)
                    # une seule fois : sinon on compte le meme graphe une
                    # fois par combinaison, soit 60 fois de trop.
                    if cond == 0 and i == 0:
                        n_graphs += sc["f1"].size
        el = time.time() - t0
        print(f"  {n_pairs} paires  ({el / n_pairs:.1f} s/paire)", flush=True)
        if max_pairs and n_pairs >= max_pairs:
            break
    return acc, n_pairs, n_graphs


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description="H10 : balayage topologique vectorise")
    p.add_argument("cache_dir")
    p.add_argument("--layer-stride", type=int, default=4)
    p.add_argument("--per-head", action="store_true",
                   help="un graphe par tete (defaut : attention moyennee)")
    p.add_argument("--keep-sinks", action="store_true",
                   help="garde BOS et le puits : la question de la tache 1")
    p.add_argument("--graphs", default=",".join(GRAPH))
    p.add_argument("--embeddings", default=",".join(EMB))
    p.add_argument("--k-values", default=",".join(map(str, K_VALUES)))
    p.add_argument("--restarts", type=int, default=5)
    p.add_argument("--backend", default="numpy", choices=["numpy", "torch"])
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--max-pairs", type=int, default=None)
    p.add_argument("--csv", default=None)
    args = p.parse_args(argv)

    graphs = tuple(args.graphs.split(","))
    embs = tuple(args.embeddings.split(","))
    ks = tuple(int(x) for x in args.k_values.split(","))
    print(f"{len(graphs)} seuillages x {len(embs)} plongements x {len(ks)} K "
          f"= {len(graphs)*len(embs)*len(ks)} combinaisons"
          f"{', puits gardes' if args.keep_sinks else ''}\n")

    acc, n_pairs, n_graphs = run(
        args.cache_dir, args.layer_stride, args.per_head, args.keep_sinks,
        args.seed, args.backend, graphs, embs, ks, args.restarts,
        args.max_pairs)
    if not acc:
        print("aucune mesure", file=sys.stderr)
        return 1

    print(f"\n{n_pairs} paires, {n_graphs} graphes par combinaison, "
          f"couches tous les {args.layer_stride}\n")
    print(f"{'graphe':<8}{'plongement':<11}{'K':>3}"
          f"{'f1 inj':>9}{'f1 ctl':>9}{'ecart':>8}"
          f"{'maj inj':>9}{'maj ctl':>9}{'ecart':>8}"
          f"{'hit inj':>9}{'hit ctl':>9}{'ecart':>8}")
    print("-" * 104)
    rows, flagged = [], []
    for g in graphs:
        for e in embs:
            for k in ks:
                a = acc.get((g, e, k))
                if a is None or a[0, 5] == 0 or a[1, 5] == 0:
                    continue
                i = a[0, :5] / a[0, 5]
                c = a[1, :5] / a[1, 5]
                print(f"{g:<8}{e:<11}{k:>3}"
                      f"{i[0]:>9.3f}{c[0]:>9.3f}{i[0]-c[0]:>+8.3f}"
                      f"{i[1]:>9.3f}{c[1]:>9.3f}{i[1]-c[1]:>+8.3f}"
                      f"{i[3]:>9.3f}{c[3]:>9.3f}{i[3]-c[3]:>+8.3f}")
                rows.append({"graph": g, "embedding": e, "k": k,
                             "f1_inj": i[0], "f1_ctl": c[0],
                             "majority_inj": i[1], "majority_ctl": c[1],
                             "contig_inj": i[2], "contig_ctl": c[2],
                             "hit_inj": i[3], "hit_ctl": c[3],
                             "f1_blind_inj": i[4], "f1_blind_ctl": c[4],
                             "n": int(a[0, 5])})
                if i[0] - c[0] > 0.05 or i[3] - c[3] > 0.05:
                    flagged.append((g, e, k, i[0] - c[0], i[3] - c[3]))
    print()
    if flagged:
        print("Combinaisons ou le clustering separe l'injection "
              "(ecart f1 ou hit > 0,05) :")
        for g, e, k, df, dh in sorted(flagged, key=lambda x: -max(x[3], x[4])):
            print(f"   {g} / {e} / K={k} : f1 {df:+.3f}, hit {dh:+.3f}")
        print("Le verdict de la serie B serait alors un artefact de "
              "construction, et cette combinaison est la piste a suivre.")
    else:
        print("Aucune combinaison ne separe l'injection : le verdict de la "
              "serie B tient, il n'etait pas un artefact du graphe choisi.")

    if args.csv:
        import csv as _csv
        os.makedirs(os.path.dirname(args.csv) or ".", exist_ok=True)
        with open(args.csv, "w", newline="", encoding="utf-8") as fh:
            w = _csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)
        print(f"\n-> {args.csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
