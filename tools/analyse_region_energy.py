"""Analyse comparative multi-modeles de l'energie regionale.

UNITE D'INFERENCE : LA PAIRE. Chaque paire fournit un Delta = C4 - C3 par
quantite. Les 21 blocs, les 36 flux et les 3 scalaires sont agreges sur les
L x H cellules a l'interieur d'une paire avant le test, parce que les
cellules d'un meme prompt ne sont pas independantes : les traiter comme
autant d'observations gonflerait artificiellement n.

    Les deux niveaux sont rendus, pas un seul.
    -- niveau PAIRE     : test et intervalle de confiance (inference)
    -- niveau CELLULE   : accord de signe et carte couche x tete (ou vit
                          l'effet). Aucun test n'est conduit a ce niveau et
                          aucune cellule n'y est selectionnee.

L'agregation intra-paire est la MOYENNE pour les quantites extensives
(energie, flux : des sommes de contributions) et elle est declaree telle.
La MEDIANE est rendue en parallele, en colonne separee, pour que la
robustesse de la conclusion a ce choix soit lisible.

TESTS. Wilcoxon apparie bilateral sur les Delta de paires, puis
Benjamini-Hochberg a q = 0,05 sur chaque famille SEPAREMENT : les 21 blocs
forment une famille, les 36 flux une autre. Intervalles de confiance a
95 % par bootstrap STRATIFIE par type d'attaque, 10 000 rejets, graine
declaree : les quatre types d'attaque n'ont pas le meme appareil (``naive``
n'a pas de preambule, ``combine`` en a un contenant deja une directive
adversariale), donc un bootstrap non stratifie melangerait des strates de
natures differentes.

DEUX BRAS DE CONTROLE. ``C4 - C3_opi`` et ``C4 - C3_wiki`` sont analyses
separement et confrontes. Une quantite n'est retenue comme signature que si
elle est significative, DE MEME SIGNE, dans les deux bras. C'est le test de
specificite au remplissage.

STRATIFICATION PAR TYPE D'ATTAQUE. Pour ``ignore`` et ``combine``, le
preambule conserve dans C3 contient deja une directive adversariale : le
contraste y isole la seule instruction de tache injectee. Les moyennes par
strate sont donc rendues, et aucune conclusion n'est tiree du seul agregat.

CE QUI N'EST PAS FAIT, ET POURQUOI. Aucun seuil n'est ajuste, aucune
cellule n'est choisie apres avoir vu les cartes, aucun classifieur n'est
entraine. La direction de l'effet n'est pas supposee : les signes sont lus,
pas predits.
"""

import argparse
import json
import os
import sys
from typing import Dict, List, Optional, Sequence

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from spectral_injection.opi import regions6 as r6

CONTROL_ARMS = ("C3_opi", "C3_wiki")
N_BOOT = 10000
BOOT_SEED = 2027
FDR_Q = 0.05


# --------------------------------------------------------------------------
# Outils statistiques
# --------------------------------------------------------------------------
def bh_fdr(p: Sequence[float]) -> np.ndarray:
    """Benjamini-Hochberg, implementation monotone decroissante."""
    p = np.asarray(p, dtype=float)
    n = p.size
    order = np.argsort(p)
    q = np.empty(n)
    run = 1.0
    for rank in range(n - 1, -1, -1):
        i = order[rank]
        run = min(run, p[i] * n / (rank + 1))
        q[i] = run
    return q


def strat_bootstrap(X: np.ndarray, strata: Sequence[str], n_boot: int,
                    seed: int) -> np.ndarray:
    """IC bootstrap stratifie. ``X`` [n, k] -> [n_boot, k] de moyennes.

    Formulation MATRICIELLE. Un rejet bootstrap est un vecteur de COMPTES
    d'occurrences sur les n paires ; la moyenne du rejet vaut donc
    ``(comptes @ X) / n``. Les ``n_boot`` rejets forment une matrice
    ``W`` [n_boot, n] et toutes les moyennes s'obtiennent en un seul
    produit, au lieu de ``n_boot`` passages en Python.

    Le tirage reste STRATIFIE : dans chaque strate on tire |strate| indices
    avec remise, independamment des autres strates. La taille de chaque
    strate est donc preservee dans tous les rejets, ce qui est le point du
    bootstrap stratifie.

    La voie matricielle suppose ``X`` fini. En presence de NaN elle
    reviendrait a traiter les manquants comme des zeros, ce qui biaiserait
    la moyenne : le code retombe alors sur la boucle avec ``nanmean``.
    """
    rng = np.random.default_rng(seed)
    n = X.shape[0]
    groups: Dict[str, List[int]] = {}
    for i, s in enumerate(strata):
        groups.setdefault(str(s), []).append(i)
    idx = [np.asarray(v) for v in groups.values()]

    take = np.concatenate(
        [g[rng.integers(0, g.size, size=(n_boot, g.size))] for g in idx],
        axis=1)                                            # [n_boot, n]

    if not np.isfinite(X).all():
        out = np.empty((n_boot,) + X.shape[1:])
        for b in range(n_boot):
            out[b] = np.nanmean(X[take[b]], axis=0)
        return out

    W = np.empty((n_boot, n))
    for b in range(n_boot):
        W[b] = np.bincount(take[b], minlength=n)
    return (W @ X) / n


def paired_test(D: np.ndarray) -> np.ndarray:
    """Wilcoxon apparie bilateral par colonne ; p = 1 si tout est nul."""
    out = np.ones(D.shape[1])
    for k in range(D.shape[1]):
        x = D[:, k]
        x = x[np.isfinite(x)]
        if x.size < 6 or not np.any(x != 0):
            continue
        try:
            out[k] = float(wilcoxon(x).pvalue)
        except ValueError:
            out[k] = 1.0
    return out


def dz(D: np.ndarray) -> np.ndarray:
    """Taille d'effet appariee ; 0 si l'ecart-type est degenere."""
    mu = np.nanmean(D, 0)
    sd = np.nanstd(D, 0, ddof=1)
    return np.where(sd > 1e-15, mu / np.where(sd > 1e-15, sd, 1.0), 0.0)


# --------------------------------------------------------------------------
# Mise en tableau
# --------------------------------------------------------------------------
def pivot_cells(long: pd.DataFrame, key_cols: List[str], value: str
                ) -> Dict[str, np.ndarray]:
    """``long`` -> {quantite: [paires, couches, tetes]} par bras.

    Rend aussi l'ordre des paires, pour que la stratification et
    l'appariement soient alignes.
    """
    long = long.copy()
    long["_q"] = long[key_cols].astype(str).agg("|".join, axis=1)
    pairs = sorted(long.pair_id.unique())
    # Garde-fou : chaque (paire, bras, couche, tete, quantite) doit
    # apparaitre UNE fois. Un identifiant de paire duplique ferait ecraser
    # une paire par l'autre dans l'affectation ci-dessous, sans erreur.
    dup = long.duplicated(["pair_id", "arm", "couche", "tete", "_q"]).sum()
    if dup:
        raise SystemExit(
            f"{dup} lignes dupliquees sur (paire, bras, couche, tete, "
            f"quantite) : identifiants de paires non uniques, le pivot "
            f"perdrait des donnees")
    pidx = {p: i for i, p in enumerate(pairs)}
    L = int(long.couche.max()) + 1
    H = int(long.tete.max()) + 1
    out = {}
    for (arm, q), g in long.groupby(["arm", "_q"], sort=True):
        A = np.full((len(pairs), L, H), np.nan)
        A[g.pair_id.map(pidx).to_numpy(),
          g.couche.to_numpy(), g.tete.to_numpy()] = g[value].to_numpy()
        out[(arm, q)] = A
    return out, pairs, L, H


def analyse_family(cells: Dict, pairs: List[str], quantities: List[str],
                   strata: Sequence[str], family: str) -> pd.DataFrame:
    """Un bras, une famille de quantites -> tableau de resultats."""
    rows = []
    for arm in CONTROL_ARMS:
        # [n, k] agreges intra-paire
        mean4 = np.stack([np.nanmean(cells[("C4", q)], axis=(1, 2))
                          for q in quantities], 1)
        mean3 = np.stack([np.nanmean(cells[(arm, q)], axis=(1, 2))
                          for q in quantities], 1)
        med4 = np.stack([np.nanmedian(cells[("C4", q)], axis=(1, 2))
                         for q in quantities], 1)
        med3 = np.stack([np.nanmedian(cells[(arm, q)], axis=(1, 2))
                         for q in quantities], 1)
        Dm = mean4 - mean3
        Dmed = med4 - med3

        p = paired_test(Dm)
        q_bh = bh_fdr(p)
        boot = strat_bootstrap(Dm, strata, N_BOOT, BOOT_SEED)
        lo = np.quantile(boot, 0.025, axis=0)
        hi = np.quantile(boot, 0.975, axis=0)
        eff = dz(Dm)

        for k, name in enumerate(quantities):
            cell_delta = (cells[("C4", name)] - cells[(arm, name)])
            med_cell = np.nanmedian(cell_delta, axis=0)          # [L, H]
            n_pos = int(np.nansum(med_cell > 0))
            n_cells = int(np.sum(np.isfinite(med_cell)))
            rows.append({
                "family": family, "arm": arm, "quantity": name,
                "delta_mean": float(np.nanmean(Dm[:, k])),
                "delta_median_agg": float(np.nanmean(Dmed[:, k])),
                "ci95_low": float(lo[k]), "ci95_high": float(hi[k]),
                "dz": float(eff[k]),
                "wilcoxon_p": float(p[k]), "bh_q": float(q_bh[k]),
                "sig_bh": bool(q_bh[k] < FDR_Q),
                "n_pairs": int(np.sum(np.isfinite(Dm[:, k]))),
                "cells_positive": n_pos, "n_cells": n_cells,
                "cell_agreement": (n_pos / n_cells) if n_cells else np.nan,
                "baseline_C3": float(np.nanmean(mean3[:, k])),
            })
    return pd.DataFrame(rows)


def concordance(tab: pd.DataFrame) -> pd.DataFrame:
    """Une quantite est-elle significative et DE MEME SIGNE dans les 2 bras ?"""
    out = []
    for (fam, q), g in tab.groupby(["family", "quantity"]):
        g = g.set_index("arm")
        if not all(a in g.index for a in CONTROL_ARMS):
            continue
        d = {a: g.loc[a, "delta_mean"] for a in CONTROL_ARMS}
        s = {a: bool(g.loc[a, "sig_bh"]) for a in CONTROL_ARMS}
        same = np.sign(d[CONTROL_ARMS[0]]) == np.sign(d[CONTROL_ARMS[1]])
        out.append({
            "family": fam, "quantity": q,
            "delta_opi": d["C3_opi"], "delta_wiki": d["C3_wiki"],
            "sig_opi": s["C3_opi"], "sig_wiki": s["C3_wiki"],
            "same_sign": bool(same),
            "robust": bool(same and s["C3_opi"] and s["C3_wiki"]),
            "agreement_opi": float(g.loc["C3_opi", "cell_agreement"]),
            "agreement_wiki": float(g.loc["C3_wiki", "cell_agreement"]),
            "min_abs_delta": float(min(abs(d["C3_opi"]), abs(d["C3_wiki"]))),
        })
    return pd.DataFrame(out).sort_values(
        "min_abs_delta", ascending=False).reset_index(drop=True)


# --------------------------------------------------------------------------
def analyse_model(d: str) -> Optional[Dict]:
    prov_p = os.path.join(d, "provenance.json")
    if not os.path.exists(prov_p):
        return None
    prov = json.load(open(prov_p, encoding="utf-8"))
    pairs_df = pd.read_csv(os.path.join(d, "pairs.csv"))
    strata_map = dict(zip(pairs_df.pair_id, pairs_df.attack_type))

    results = []

    blocks = pd.read_parquet(os.path.join(d, "blocks_long.parquet"))
    cells, pairs, L, H = pivot_cells(blocks, ["bloc"], "energy")
    strata = [strata_map[p] for p in pairs]
    qs = sorted({k[1] for k in cells})
    results.append(analyse_family(cells, pairs, qs, strata, "block_energy"))

    cells_s, pairs_s, _, _ = pivot_cells(blocks, ["bloc"], "share")
    results.append(analyse_family(cells_s, pairs_s, qs,
                                  [strata_map[p] for p in pairs_s],
                                  "block_share"))
    del blocks, cells, cells_s

    flows = pd.read_parquet(os.path.join(d, "flows_long.parquet"))
    cells_f, pairs_f, _, _ = pivot_cells(flows, ["query", "key"], "flow")
    qf = sorted({k[1] for k in cells_f})
    results.append(analyse_family(cells_f, pairs_f, qf,
                                  [strata_map[p] for p in pairs_f],
                                  "directed_flow"))
    del flows, cells_f

    scal = pd.read_parquet(os.path.join(d, "scalars_long.parquet"))
    sl = scal.melt(id_vars=["pair_id", "arm", "couche", "tete"],
                   value_vars=["offdiag", "loop", "variance"],
                   var_name="scalar", value_name="v")
    cells_x, pairs_x, _, _ = pivot_cells(sl, ["scalar"], "v")
    results.append(analyse_family(cells_x, pairs_x,
                                  sorted({k[1] for k in cells_x}),
                                  [strata_map[p] for p in pairs_x], "scalar"))
    del scal, sl, cells_x

    tab = pd.concat(results, ignore_index=True)
    tab.insert(0, "model", prov["tag"])
    conc = concordance(tab)
    conc.insert(0, "model", prov["tag"])

    tab.to_csv(os.path.join(d, "analysis_tests.csv"), index=False)
    conc.to_csv(os.path.join(d, "analysis_concordance.csv"), index=False)
    return {"prov": prov, "tests": tab, "conc": conc,
            "n_pairs": len(pairs), "L": L, "H": H,
            "pairs_df": pairs_df}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=os.path.join("results", "opi",
                                                   "region_energy"))
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    out = args.out or args.root
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

    tags = sorted(t for t in os.listdir(args.root)
                  if os.path.isdir(os.path.join(args.root, t))
                  and not t.startswith("_"))
    models = []
    for t in tags:
        r = analyse_model(os.path.join(args.root, t))
        if r is None:
            print(f"  {t} : incomplet, ignore")
            continue
        models.append(r)
        print(f"  {t} : {r['n_pairs']} paires, {r['L']}x{r['H']} cellules")
    if not models:
        raise SystemExit("aucun modele exploitable")

    tests = pd.concat([m["tests"] for m in models], ignore_index=True)
    conc = pd.concat([m["conc"] for m in models], ignore_index=True)
    tests.to_csv(os.path.join(out, "all_tests.csv"), index=False)
    conc.to_csv(os.path.join(out, "all_concordance.csv"), index=False)

    # -- reproductibilite inter-modeles : une quantite robuste dans combien
    #    de modeles, et de quel signe ?
    rep = []
    for (fam, q), g in conc.groupby(["family", "quantity"]):
        rob = g[g.robust]
        signs = np.sign(rob.delta_opi.to_numpy())
        rep.append({
            "family": fam, "quantity": q,
            "n_models": int(len(g)),
            "n_robust": int(len(rob)),
            "n_positive": int((signs > 0).sum()),
            "n_negative": int((signs < 0).sum()),
            "unanimous": bool(len(rob) == len(g) and len(g) > 1
                              and len(set(signs.tolist())) == 1),
            "median_delta_opi": float(g.delta_opi.median()),
            "median_delta_wiki": float(g.delta_wiki.median()),
            "median_agreement": float(
                g[["agreement_opi", "agreement_wiki"]].mean(axis=1).median()),
        })
    rep = pd.DataFrame(rep).sort_values(
        ["unanimous", "n_robust", "median_agreement"],
        ascending=[False, False, False]).reset_index(drop=True)
    rep.to_csv(os.path.join(out, "reproducibility.csv"), index=False)

    print(f"\n{len(models)} modeles, {len(tests)} tests, "
          f"{len(rep)} quantites distinctes")
    print(f"ecrits : all_tests.csv, all_concordance.csv, reproducibility.csv")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
