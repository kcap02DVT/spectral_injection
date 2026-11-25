"""Cloture de la tache 1 : lambda_2 detecte-t-il sur G1-Kron ?

La tache 1 a montre que la suppression seche du puits fait tomber lambda_2
de 72 % et fait sauter le vecteur de Fiedler sur une coupe sans rapport
(alignement 0,58), alors que la reduction de Kron le laisse intact
(alignement 0,96). Il reste a savoir ce que ca fait a la *detection*.

Le plan experimental est le plus propre possible, et c'est le point : les
quatre topologies sont calculees sur les memes paires, aux memes couches,
aux memes tetes, dans les memes deux conditions. Rien ne varie d'une
colonne a l'autre que la construction du graphe. Il n'y a donc pas de
transfert a faire ni de nouvelle correction a payer : une cellule choisie
sur G2 se lit telle quelle sur G1-Kron.

    g2        attention brute symetrisee, puits et template gardes
    g1_drop   puits supprimes, lignes et colonnes retirees  (le G1 actuel)
    g1_renorm supprimes, degres des noeuds gardes restaures
    g1_kron   complement de Schur sur le bloc garde (resistances preservees)

Le puits est detecte une fois par couche sur le graphe moyenne, pas tete par
tete : sinon les quatorze tetes n'auraient pas le meme support et la
decomposition propre ne serait plus batchable -- et surtout les cellules ne
seraient plus comparables entre topologies, ce qui est tout l'objet.

Les cinq metriques de spectral-trust sont recalculees ici a l'identique
(lambda_2, lambda_2/lambda_max, entropie spectrale normalisee, hfer,
lambda_max), a partir des seules valeurs propres : ``eigvalsh`` batche sur
les tetes suffit, aucun vecteur propre n'est necessaire.

Sortie : pour chaque topologie, le nombre de cellules qui survivent a la
correction familiale, la meilleure AUROC, et -- le chiffre qui tranche --
l'AUROC des cellules choisies sur G2 relues sur chaque topologie. Plus la
matrice de correlation des cinq metriques, qui chiffre le paradoxe 3.

    python -m spectral_injection.lambda2_topo cache/attn/MODELE/SPLIT \\
        --layer-stride 4
"""

import argparse
import os
import sys
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from .prompts import ROLE_INJECTION, ROLE_TEMPLATE

EPS = 1e-12
METRICS = ["fiedler_value", "connectivity_ratio", "spectral_entropy_norm",
           "hfer", "spectral_radius"]
TOPOLOGIES = ["g2", "g1_drop", "g1_renorm", "g1_kron"]


# --------------------------------------------------------------------------
# Topologies, batchees sur les tetes
# --------------------------------------------------------------------------
def symmetrize_heads(attn_layer: np.ndarray) -> np.ndarray:
    """[H,T,T] -> W symetrique par tete, diagonale nulle."""
    A = np.asarray(attn_layer, dtype=np.float64)
    W = 0.5 * (A + np.transpose(A, (0, 2, 1)))
    idx = np.arange(W.shape[-1])
    W[:, idx, idx] = 0.0
    return W


def sinks_from_mean(W: np.ndarray, share: float = 0.02) -> np.ndarray:
    """Puits determines sur la moyenne des tetes : un support commun.

    BOS, l'argmax de masse recue, et tout noeud au-dela de ``share`` de la
    masse totale -- la meme regle que ``sinks.sink_indices``, appliquee une
    fois pour toutes les tetes de la couche.
    """
    m = W.mean(0)
    mass = m.sum(0)
    total = max(mass.sum(), EPS)
    idx = set(np.flatnonzero(mass / total >= share).tolist())
    idx.add(int(mass.argmax()))
    idx.add(0)
    return np.array(sorted(idx), dtype=int)


def topo_drop(W: np.ndarray, S: np.ndarray, K: np.ndarray) -> np.ndarray:
    return W[:, K][:, :, K]


def topo_renorm(W: np.ndarray, S: np.ndarray, K: np.ndarray) -> np.ndarray:
    """Degres des noeuds gardes restaures, symetriquement."""
    deg_before = W.sum(-1)[:, K]
    sub = W[:, K][:, :, K]
    deg_after = sub.sum(-1)
    f = np.where(deg_after > EPS, deg_before / np.maximum(deg_after, EPS), 1.0)
    return sub * np.sqrt(f[:, :, None] * f[:, None, :])


def topo_kron(W: np.ndarray, S: np.ndarray, K: np.ndarray) -> np.ndarray:
    """Complement de Schur du laplacien combinatoire, batche sur les tetes."""
    H, n, _ = W.shape
    d = W.sum(-1)
    L = -W.copy()
    idx = np.arange(n)
    L[:, idx, idx] += d
    L_kk = L[:, K][:, :, K]
    L_ks = L[:, K][:, :, S]
    L_ss = L[:, S][:, :, S]
    red = L_kk - L_ks @ np.linalg.pinv(L_ss) @ np.transpose(L_ks, (0, 2, 1))
    red = 0.5 * (red + np.transpose(red, (0, 2, 1)))
    Wr = -red
    k = np.arange(Wr.shape[-1])
    Wr[:, k, k] = 0.0
    return np.clip(Wr, 0.0, None)


BUILDERS = {"g1_drop": topo_drop, "g1_renorm": topo_renorm,
            "g1_kron": topo_kron}


# --------------------------------------------------------------------------
# Les cinq metriques, a partir des valeurs propres seules
# --------------------------------------------------------------------------
def metrics_from_W(W: np.ndarray) -> np.ndarray:
    """W [H,n,n] -> [H,5], memes definitions que spectral_trust 0.3.0."""
    d = W.sum(-1)
    inv = np.where(d > EPS, 1.0 / np.sqrt(np.maximum(d, EPS)), 0.0)
    L = -W * inv[:, :, None] * inv[:, None, :]
    idx = np.arange(W.shape[-1])
    L[:, idx, idx] += 1.0
    L = 0.5 * (L + np.transpose(L, (0, 2, 1)))
    ev = np.clip(np.linalg.eigvalsh(L), 0.0, None)          # [H,n], croissant
    n = ev.shape[-1]
    lam2 = ev[:, 1]
    lmax = ev[:, -1]
    total = ev.sum(-1)
    p = np.clip(ev / np.maximum(total, EPS)[:, None], EPS, None)
    ent = -(p * np.log(p)).sum(-1) / np.log(n)
    hfer = ev[:, n // 2:].sum(-1) / np.maximum(total, EPS)
    conn = np.where(lmax > EPS, lam2 / np.maximum(lmax, EPS), 0.0)
    return np.stack([lam2, conn, ent, hfer, lmax], axis=-1)


def layer_values(attn_layer: np.ndarray, roles: np.ndarray,
                 share: float = 0.02) -> Dict[str, np.ndarray]:
    """Une couche -> {topologie: [H,5]}."""
    W = symmetrize_heads(attn_layer)
    keep_t = roles != ROLE_TEMPLATE
    W = W[:, keep_t][:, :, keep_t]
    n = W.shape[-1]
    S = sinks_from_mean(W, share)
    K = np.setdiff1d(np.arange(n), S)
    out = {"g2": metrics_from_W(W)}
    for name, fn in BUILDERS.items():
        out[name] = metrics_from_W(fn(W, S, K))
    return out


# --------------------------------------------------------------------------
# Campagne
# --------------------------------------------------------------------------
def run(cache_dir: str, layer_stride: int = 4, share: float = 0.02,
        max_pairs: Optional[int] = None
        ) -> Tuple[Dict[str, np.ndarray], List[int]]:
    """Rend {topologie: V [N,2,L,H,5]} et la liste des couches."""
    from .attn_cache import iter_cache
    import time

    acc: Dict[str, List[np.ndarray]] = {t: [] for t in TOPOLOGIES}
    layers: List[int] = []
    n = 0
    t0 = time.time()
    for att_i, att_c, roles_i, roles_c, meta in iter_cache(cache_dir):
        n += 1
        ls = list(range(0, att_i.shape[0], layer_stride))
        layers = ls
        per_cond = {t: [] for t in TOPOLOGIES}
        for att, roles in ((att_i, roles_i), (att_c, roles_c)):
            per_layer = {t: [] for t in TOPOLOGIES}
            for l in ls:
                v = layer_values(att[l], roles, share)
                for t in TOPOLOGIES:
                    per_layer[t].append(v[t])
            for t in TOPOLOGIES:
                per_cond[t].append(np.stack(per_layer[t]))      # [L,H,5]
        for t in TOPOLOGIES:
            acc[t].append(np.stack(per_cond[t]))                # [2,L,H,5]
        print(f"  {n} paires  ({(time.time()-t0)/n:.1f} s/paire)", flush=True)
        if max_pairs and n >= max_pairs:
            break
    return {t: np.stack(v) for t, v in acc.items()}, layers


# --------------------------------------------------------------------------
# Analyse
# --------------------------------------------------------------------------
def cell_index(L: int, H: int, M: int):
    """Indices (couche, tete, metrique) pour un tenseur aplati."""
    lidx = np.repeat(np.arange(L), H * M)
    hidx = np.tile(np.repeat(np.arange(H), M), L)
    midx = np.tile(np.arange(M), L * H)
    return lidx, hidx, midx


def paired_auroc(V: np.ndarray, cells: np.ndarray, sign: np.ndarray
                 ) -> np.ndarray:
    """AUROC injecte contre controle, par cellule, score oriente."""
    from .hypotheses_report import auroc

    N = V.shape[0]
    fi = V[:, 0].reshape(N, -1)
    fc = V[:, 1].reshape(N, -1)
    y = np.r_[np.ones(N), np.zeros(N)]
    out = np.full(cells.size, np.nan)
    for i, c in enumerate(cells):
        s = np.r_[fi[:, c], fc[:, c]] * sign[i]
        if np.isfinite(s).all():
            out[i] = auroc(y, s)
    return out


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description="lambda_2 sur G2 / G1-drop / G1-Kron")
    p.add_argument("cache_dir")
    p.add_argument("--layer-stride", type=int, default=4)
    p.add_argument("--share", type=float, default=0.02)
    p.add_argument("--max-pairs", type=int, default=None)
    p.add_argument("--top", type=int, default=10)
    p.add_argument("--out", default=None)
    args = p.parse_args(argv)

    from .specificity import family_stats

    Vs, layers = run(args.cache_dir, args.layer_stride, args.share,
                     args.max_pairs)
    N, _, L, H, M = Vs["g2"].shape
    lidx, hidx, midx = cell_index(L, H, M)
    print(f"\n{N} paires · {L} couches (pas {args.layer_stride}) · {H} tetes "
          f"· {M} metriques = {L*H*M} cellules par topologie\n")

    # ---- par topologie : combien de cellules tiennent, et la meilleure
    stats, D = {}, {}
    for t in TOPOLOGIES:
        d = (Vs[t][:, 0] - Vs[t][:, 1]).reshape(N, -1)
        fam = np.isfinite(d).all(0)
        st = family_stats(d[:, fam], n_perm=5000, seed=0)
        stats[t] = (st, np.flatnonzero(fam))
        D[t] = d

    print("Selection independante par topologie "
          "(correction familiale par inversion de signe, 5000 tirages)")
    print(f"{'topologie':<11}{'cellules':>9}{'p_fwer<0.05':>13}"
          f"{'max |dz|':>10}{'meilleure cellule':>34}")
    print("-" * 77)
    for t in TOPOLOGIES:
        st, cells = stats[t]
        keep = st["p_fwer"] < 0.05
        dz = st["dz"]
        j = int(np.argmax(np.abs(dz)))
        c = cells[j]
        print(f"{t:<11}{cells.size:>9}{int(keep.sum()):>13}{abs(dz[j]):>10.3f}"
              f"{METRICS[midx[c]]+' c'+str(layers[lidx[c]])+' t'+str(hidx[c]):>34}")

    # ---- le chiffre qui tranche : cellules choisies sur G2, relues ailleurs
    st, cells = stats["g2"]
    keep = st["p_fwer"] < 0.05
    n_sig = int(keep.sum())
    if n_sig == 0:
        # Avec N paires il n'existe que 2^N inversions de signe, donc le p
        # familial ne peut pas descendre sous 2^-N : a N = 3 le minimum est
        # 0,125 et aucune cellule ne peut etre significative, quelle que
        # soit la force de l'effet. On retombe sur le classement par |dz|
        # en le disant, plutot que de rendre un tableau vide.
        print(f"\nATTENTION : aucune cellule ne passe la correction familiale "
              f"(N = {N} paires, p familial minimal 2^-N = {2.0**-N:.3f}).")
        print("Les cellules ci-dessous sont classees par |dz| seul : ce sont "
              "des directions, pas des resultats.")
        sel, dz = cells, st["dz"]
    else:
        sel, dz = cells[keep], st["dz"][keep]
    order = np.argsort(-np.abs(dz))[:args.top]
    sel, dz = sel[order], dz[order]
    sign = np.sign(dz)
    print(f"\nCellules choisies sur G2 ({sel.size} retenues sur "
          f"{n_sig} significatives), relues telles quelles :")
    print(f"{'topologie':<11}{'AUROC moy':>11}{'AUROC max':>11}"
          f"{'dz moyen':>11}{'cellules de meme signe':>24}")
    print("-" * 68)
    base = None
    for t in TOPOLOGIES:
        a = paired_auroc(Vs[t], sel, sign)
        d = D[t][:, sel] * sign[None, :]
        dzt = d.mean(0) / np.maximum(d.std(0, ddof=1), EPS)
        same = int((dzt > 0).sum())
        print(f"{t:<11}{np.nanmean(a):>11.3f}{np.nanmax(a):>11.3f}"
              f"{dzt.mean():>11.3f}{f'{same}/{sel.size}':>24}")
        if t == "g2":
            base = a

    # ---- detail des cellules retenues
    print(f"\nDetail des {sel.size} cellules (orientees par le signe sur G2)")
    print(f"{'metrique':<22}{'couche':>7}{'tete':>5}"
          + "".join(f"{t:>11}" for t in TOPOLOGIES))
    print("-" * 78)
    aur = {t: paired_auroc(Vs[t], sel, sign) for t in TOPOLOGIES}
    for i, c in enumerate(sel):
        print(f"{METRICS[midx[c]]:<22}{layers[lidx[c]]:>7}{hidx[c]:>5}"
              + "".join(f"{aur[t][i]:>11.3f}" for t in TOPOLOGIES))

    # ---- paradoxe 3 : redondance des cinq metriques
    print("\nRedondance des cinq metriques sur G2 "
          "(correlation de Pearson des dz, par cellule)")
    st2, cells2 = stats["g2"]
    dzg2 = np.full(L * H * M, np.nan)
    dzg2[cells2] = st2["dz"]
    bym = [dzg2[midx == m] for m in range(M)]
    print(" " * 22 + "".join(f"{n[:9]:>11}" for n in METRICS))
    for a in range(M):
        row = f"{METRICS[a]:<22}"
        for b in range(M):
            ok = np.isfinite(bym[a]) & np.isfinite(bym[b])
            r = (np.corrcoef(bym[a][ok], bym[b][ok])[0, 1]
                 if ok.sum() > 2 else np.nan)
            row += f"{r:>11.3f}"
        print(row)

    print("\nLecture. Si G1-Kron garde l'AUROC de G2 et G1-drop la perd, le")
    print("paradoxe de la tache 1 est clos : ce n'est pas le retrait du puits")
    print("qui detruit le signal, c'est la facon de le retirer. Les cellules")
    print("etant identiques et les paires les memes, rien d'autre ne varie.")

    if args.out:
        os.makedirs(args.out, exist_ok=True)
        np.savez_compressed(os.path.join(args.out, "values.npz"),
                            layers=np.array(layers),
                            metrics=np.array(METRICS),
                            topologies=np.array(TOPOLOGIES),
                            **{t: Vs[t] for t in TOPOLOGIES})
        print(f"\n-> {args.out}/values.npz")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
