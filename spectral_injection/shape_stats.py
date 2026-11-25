"""Statistiques de forme du profil d'energie d'ondelette.

PRE-ENREGISTREMENT, arrete le 3 novembre 2026, AVANT toute execution.

Motivation. La recherche de la pire fenetre a ete qualifiee hors echantillon
sur la cellule couche 8 tete 7 (apparie 0,650, p = 0,040) mais le mur de
localisation persiste : IoU 0,049, top-k 0,175. L'hypothese de rechange est
qu'une injection n'a pas besoin d'etre l'anomalie dominante pour deformer la
*distribution* des energies du prompt. Ces quatre statistiques resument le
profil entier au lieu d'en chercher l'extremum.

Les quatre metriques, une valeur par echelle, calculees sur E [n_echelles, T] :

    iqr        Q75(E) - Q25(E). Dispersion globale.
    skew       asymetrie standardisee de Fisher, m3 / sd^3. Negative quand la
               queue gauche -- les creux -- est la plus lourde.
    low_mass   part des noeuds sous un seuil d'energie bas.
    ipr        somme(E^2) / somme(E)^2. Concentration spectrale.

DEUX DECISIONS DE SPECIFICATION, prises ici et pas apres coup.

1.  ``low_mass`` ne peut pas etre << la part des noeuds sous le 10e centile
    de la distribution du graphe >> : cette quantite vaut 0,10 pour tout
    prompt, par definition du centile. Elle serait une constante et ne
    mesurerait rien. Le seuil doit venir d'un modele de reference, pas de
    l'echantillon lui-meme. Retenu :

        thr = mediane - 1,2816 x MAD_normalisee

    soit le 10e centile d'une gaussienne de meme position et meme echelle.
    ``low_mass`` mesure donc l'*exces* de masse dans la queue basse par
    rapport a une gaussienne equivalente -- une vraie statistique de forme,
    non degeneree, et qui dit exactement ce que l'hypothese annonce.
    La variante ``low_mass_halfmed`` (part des noeuds sous la moitie de la
    mediane) est calculee aussi : meme intention, seuil plus lisible, aucune
    hypothese de forme.

2.  ``ipr`` tel qu'ecrit depend de T : pour un profil uniforme il vaut 1/T.
    Il melange donc concentration et longueur de prompt. A l'interieur d'une
    paire ce n'est pas un probleme -- le controle a, par construction, la
    meme longueur totale que l'attaque -- mais l'AUROC *pooled* compare des
    prompts de longueurs differentes et le confond. ``ipr`` brut est garde
    comme demande, et ``ipr_norm = T x somme(E^2)/somme(E)^2`` est ajoute :
    il vaut 1 pour un profil uniforme, quelle que soit la longueur.

REGLE DE SELECTION, fixee d'avance.

La famille primaire est 4 metriques x 5 echelles = 20 candidats, a la seule
cellule pre-enregistree (couche 8, tete 7). La metrique-echelle de plus
grande AUROC pooled sur ``train`` devient le candidat unique lu une seule
fois sur ``test``.

Avec 20 candidats et n = 37, le maximum d'AUROC sous hypothese nulle est
nettement au-dessus de 0,5 : il est donc calcule, par permutation des
etiquettes a l'interieur des paires, et rapporte a cote du maximum observe.
Un maximum qui ne depasse pas ce seuil ne designe aucun candidat, et la
lecture sur ``test`` n'aura pas lieu -- c'est la condition d'arret, posee
avant d'avoir vu un chiffre.

Les variantes (``low_mass_halfmed``, ``ipr_norm``, ``iqr_rel``) et les autres
tetes de la couche 8 sont calculees et rapportees comme EXPLORATOIRES. Elles
ne peuvent pas devenir le candidat de test : les inclure ferait passer la
famille de 20 a plusieurs centaines et viderait le pre-enregistrement de son
sens.

    python -m spectral_injection.shape_stats cache/attn/MODELE/SPLIT \\
        --layer 8 --head 7
"""

import argparse
import csv
import os
import sys
from typing import Dict, List, Optional, Sequence

import numpy as np

from .prompts import ROLE_INJECTION
from .wavelets import graph_from_attention
from .wavelet_zscore import SCALES, multiscale_wavelet_profile

EPS = 1e-12

#: Les quatre metriques pre-enregistrees. L'ordre est celui de l'enonce.
PRIMARY = ["iqr", "skew", "low_mass", "ipr"]

#: SECOND PRE-ENREGISTREMENT, arrete le 3 novembre 2026, apres la lecture du
#: train pour les formes brutes et AVANT toute execution de la version
#: standardisee. Le split test reste verrouille et non consulte.
#:
#: Constat qui motive : sur train, les formes brutes donnent un apparie de
#: 0,757 (28/37) pour un pooled de 0,606 au mieux -- ipr va jusqu'a 0,757
#: apparie pour 0,508 pooled. Le signal est reel mais enferme dans une
#: reference propre a chaque document, exactement le gouffre que le z-score
#: intra-prompt avait comble pour l'energie par fenetre.
#:
#: Standardisation retenue, par prompt et par echelle :
#:
#:     z = (stat_obs - moyenne du nul) / ecart-type du nul
#:
#: le nul etant des surrogates a phases randomisees -- meme spectre de
#: puissance, donc meme autocorrelation et meme variance (Parseval), mais
#: aucune structure localisee. z mesure donc, metrique par metrique, l'ecart
#: a ce qu'un profil de meme structure au second ordre produirait. C'est une
#: reference INTERNE au prompt : rien n'est compare entre documents, et le
#: niveau de base propre au document disparait.
#:
#: Famille : 4 metriques standardisees x 5 echelles = 20 candidats, a la
#: seule cellule couche 8 tete 7. Selection par AUROC pooled maximale.
#: Condition d'arret inchangee : si le maximum observe ne depasse pas le q95
#: du nul par permutation intra-paires, aucun candidat n'est designe.
STANDARDIZED = ["z_iqr", "z_skew", "z_low_mass", "z_ipr"]

#: Variantes et diagnostics -- exploratoires, non eligibles au test.
EXPLORATORY = ["iqr_rel", "low_mass_halfmed", "ipr_norm"]

NAMES = PRIMARY + EXPLORATORY + STANDARDIZED

#: Direction attendue, +1 = la valeur monte sous injection. Pre-enregistree.
PREDICTED = {
    "iqr": +1,          # la queue basse s'etire, la dispersion augmente
    "skew": -1,         # un creux alourdit la queue gauche
    "low_mass": +1,     # plus de noeuds sous le seuil bas
    "ipr": +1,          # l'energie se concentre hors du mur lisse
    "iqr_rel": +1, "low_mass_halfmed": +1, "ipr_norm": +1,
    # Les versions standardisees heritent de la direction de leur base.
    "z_iqr": +1, "z_skew": -1, "z_low_mass": +1, "z_ipr": +1,
}


def global_shape_statistics(E: np.ndarray) -> Dict[str, np.ndarray]:
    """Profil ``[n_echelles, T]`` -> une valeur par echelle et par metrique.

    Aucune etiquette, aucun apprentissage, aucune fenetre : chaque statistique
    resume la distribution des energies du prompt entier.
    """
    E = np.asarray(E, dtype=np.float64)
    if E.ndim != 2:
        raise ValueError(f"attendu [n_echelles, T], recu {E.shape}")
    T = E.shape[1]

    q25 = np.quantile(E, 0.25, axis=1)
    q75 = np.quantile(E, 0.75, axis=1)
    med = np.median(E, axis=1)
    mu = E.mean(1)
    sd = E.std(1, ddof=1)
    mad = np.median(np.abs(E - med[:, None]), axis=1) * 1.4826

    m3 = ((E - mu[:, None]) ** 3).mean(1)
    skew = np.where(sd > EPS, m3 / np.maximum(sd ** 3, EPS), 0.0)

    # 10e centile d'une gaussienne de meme position et meme echelle : le
    # seuil ne vient pas de l'echantillon, donc la quantite n'est pas
    # constante par construction.
    thr = med - 1.2815515655446004 * mad
    low = (E < thr[:, None]).mean(1)
    low_half = (E < 0.5 * med[:, None]).mean(1)

    s1 = E.sum(1)
    s2 = (E ** 2).sum(1)
    ipr = s2 / np.maximum(s1 ** 2, EPS)

    return {
        "iqr": q75 - q25,
        "skew": skew,
        "low_mass": low,
        "ipr": ipr,
        "iqr_rel": (q75 - q25) / np.maximum(med, EPS),
        "low_mass_halfmed": low_half,
        "ipr_norm": T * ipr,
    }


def _phase_surrogate_batch(E: np.ndarray, rng) -> np.ndarray:
    """Surrogates a spectre de puissance conserve, une phase par echelle.

    |FFT| est garde, les phases sont tirees au hasard. L'autocorrelation et
    la variance du profil sont donc preservees exactement (Parseval), et
    toute structure localisee est detruite. Chaque echelle recoit ses propres
    phases, parce que les echelles sont des signaux distincts.
    """
    out = np.empty_like(E)
    for k in range(E.shape[0]):
        x = E[k]
        F = np.fft.rfft(x - x.mean())
        ph = rng.uniform(0.0, 2.0 * np.pi, F.size)
        ph[0] = 0.0
        if x.size % 2 == 0:
            ph[-1] = 0.0
        out[k] = np.fft.irfft(np.abs(F) * np.exp(1j * ph), n=x.size) + x.mean()
    return out


def surrogate_shape_z(E: np.ndarray, n_surrogates: int = 200,
                      seed: int = 0) -> Dict[str, np.ndarray]:
    """Les quatre formes, standardisees par leur propre nul de surrogates.

    Rend ``{z_iqr, z_skew, z_low_mass, z_ipr}``, un vecteur par echelle.

    Un seul jeu de surrogates sert aux quatre metriques : c'est voulu, ca
    preserve la structure jointe du nul au lieu de la casser, et ca divise le
    cout par quatre.

    Ce que chaque z devient. La randomisation de phase gaussianise le profil
    en gardant sa variance :

        z_iqr       exces de dispersion interquartile sur ce qu'une
                    gaussienne de meme variance donnerait
        z_skew      asymetrie standardisee, le nul etant centre sur 0
        z_low_mass  exces de masse dans la queue basse -- c'est la
                    definition que ``low_mass`` cherchait, enfin calibree
                    sans seuil gaussien pose a la main
        z_ipr       exces de concentration de l'energie
    """
    E = np.asarray(E, dtype=np.float64)
    obs = global_shape_statistics(E)
    rng = np.random.default_rng(seed)
    acc = {n: np.empty((n_surrogates, E.shape[0])) for n in PRIMARY}
    for b in range(n_surrogates):
        st = global_shape_statistics(_phase_surrogate_batch(E, rng))
        for n in PRIMARY:
            acc[n][b] = st[n]
    out = {}
    for n in PRIMARY:
        mu = acc[n].mean(0)
        sd = acc[n].std(0, ddof=1)
        out[f"z_{n}"] = (obs[n] - mu) / np.maximum(sd, EPS)
    return out


# --------------------------------------------------------------------------
# Campagne
# --------------------------------------------------------------------------
def run(cache_dir: str, layer: int = 8, heads: Optional[Sequence[int]] = None,
        scales: Sequence[float] = SCALES, reduction: str = "kron",
        sparse: str = "q90", kernel: str = "mexican_hat",
        max_pairs: Optional[int] = None,
        n_surrogates: int = 200) -> List[Dict]:
    """Un enregistrement par (paire, tete, condition), toutes metriques."""
    from .attn_cache import iter_cache
    import time

    rows: List[Dict] = []
    n = 0
    t0 = time.time()
    for att_i, att_c, roles_i, roles_c, meta in iter_cache(cache_dir):
        n += 1
        hs = list(range(att_i.shape[1])) if heads is None else list(heads)
        for h in hs:
            for cond, att, roles in (("injected", att_i, roles_i),
                                     ("control", att_c, roles_c)):
                L, r, pos = graph_from_attention(att[layer], roles, reduction,
                                                 head=h, sparse=sparse)
                E = multiscale_wavelet_profile(L, scales, kernel)
                st = global_shape_statistics(E)
                if n_surrogates:
                    st.update(surrogate_shape_z(E, n_surrogates, 0))
                row = {"pair": int(meta["pair"]), "layer": int(layer),
                       "head": int(h), "condition": cond,
                       "category": meta["category"], "n_nodes": int(L.shape[0])}
                for name in NAMES:
                    if name not in st:
                        continue
                    for k, s in enumerate(scales):
                        row[f"{name}@{s:g}"] = float(st[name][k])
                rows.append(row)
        print(f"  {n} paires  ({(time.time()-t0)/n:.1f} s/paire)", flush=True)
        if max_pairs and n >= max_pairs:
            break
    return rows


def _auroc(y: np.ndarray, sc: np.ndarray) -> float:
    from .hypotheses_report import auroc
    return auroc(y, sc)


def evaluate(rows: List[Dict], head: int, scales: Sequence[float],
             names: Sequence[str], n_perm: int = 2000, seed: int = 0,
             family: Optional[Sequence[str]] = None) -> Dict[str, object]:
    """AUROC pooled et appariee par candidat, plus le maximum sous le nul.

    Le nul permute l'etiquette injecte/controle *a l'interieur de chaque
    paire* : c'est l'echangeabilite qui tient sous l'hypothese nulle d'un
    plan apparie, et elle preserve la correlation entre candidats, donc le
    maximum sur la famille est un seuil familial legitime.

    ``family`` designe les candidats eligibles a la selection et au calcul du
    nul. Les autres sont evalues mais ne peuvent pas devenir le candidat.
    """
    sub = [r for r in rows if r["head"] == head]
    inj = {r["pair"]: r for r in sub if r["condition"] == "injected"}
    ctl = {r["pair"]: r for r in sub if r["condition"] == "control"}
    keys = sorted(set(inj) & set(ctl))
    n = len(keys)
    y = np.r_[np.ones(n), np.zeros(n)]

    cands, A, B = [], [], []
    for name in names:
        for s_ in scales:
            col = f"{name}@{s_:g}"
            if col not in inj[keys[0]]:
                continue
            a = np.array([inj[k][col] for k in keys], dtype=float)
            b = np.array([ctl[k][col] for k in keys], dtype=float)
            if not (np.isfinite(a).all() and np.isfinite(b).all()):
                continue
            cands.append((name, float(s_)))
            A.append(a)
            B.append(b)
    A, B = np.array(A), np.array(B)

    out = []
    for i, (name, s_) in enumerate(cands):
        sign = PREDICTED.get(name, 1)
        a, b = A[i] * sign, B[i] * sign
        pooled = _auroc(y, np.r_[a, b])
        out.append({"metric": name, "scale": s_, "n": n,
                    "auroc_pooled": pooled,
                    "auroc_pooled_abs": max(pooled, 1 - pooled),
                    "auroc_paired": float((a > b).mean()),
                    "wins": int((a > b).sum())})

    fam = list(family) if family is not None else PRIMARY
    prim = [i for i, (nm, _) in enumerate(cands) if nm in fam]
    rng = np.random.default_rng(seed)
    null_max = np.empty(n_perm)
    for q in range(n_perm):
        flip = rng.random(n) < 0.5
        best = 0.0
        for i in prim:
            sign = PREDICTED.get(cands[i][0], 1)
            a, b = A[i] * sign, B[i] * sign
            aa = np.where(flip, b, a)
            bb = np.where(flip, a, b)
            v = _auroc(y, np.r_[aa, bb])
            best = max(best, max(v, 1 - v))
        null_max[q] = best
    return {"rows": out, "n": n, "n_primary": len(prim),
            "null_max_mean": float(null_max.mean()),
            "null_max_q95": float(np.quantile(null_max, 0.95)),
            "null_max": null_max}


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description="statistiques de forme du profil")
    p.add_argument("cache_dir")
    p.add_argument("--layer", type=int, default=8)
    p.add_argument("--head", type=int, default=7,
                   help="cellule pre-enregistree pour l'evaluation primaire")
    p.add_argument("--all-heads", action="store_true",
                   help="calcule toutes les tetes de la couche (exploratoire)")
    p.add_argument("--scales", default=",".join(str(s) for s in SCALES))
    p.add_argument("--reduction", default="kron")
    p.add_argument("--sparse", default="q90")
    p.add_argument("--n-perm", type=int, default=2000)
    p.add_argument("--surrogates", type=int, default=200,
                   help="surrogates de phase par prompt pour la "
                        "standardisation (0 = formes brutes seules)")
    p.add_argument("--max-pairs", type=int, default=None)
    p.add_argument("--csv", default=None)
    args = p.parse_args(argv)

    scales = tuple(float(x) for x in args.scales.split(","))
    heads = None if args.all_heads else [args.head]
    rows = run(args.cache_dir, args.layer, heads, scales, args.reduction,
               args.sparse, max_pairs=args.max_pairs,
               n_surrogates=args.surrogates)
    if not rows:
        print("aucune mesure", file=sys.stderr)
        return 1

    print()
    print(f"couche {args.layer}, tete {args.head} (PRE-ENREGISTREE), "
          f"{args.reduction} + {args.sparse}, "
          f"{args.surrogates} surrogates par prompt")
    print()

    for label, fam, is_primary in (
            ("FORMES BRUTES (premier pre-enregistrement)", PRIMARY, False),
            ("FORMES STANDARDISEES par surrogates "
             "(second pre-enregistrement)", STANDARDIZED, True)):
        res = evaluate(rows, args.head, scales, NAMES, args.n_perm,
                       family=fam)
        sel = [r for r in res["rows"] if r["metric"] in fam]
        if not sel:
            continue
        print(f"{label} -- {len(sel)} candidats, n = {res['n']} paires")
        print(f"{'metrique':<18}{'echelle':>9}{'AUROC pooled':>14}"
              f"{'AUROC apparie':>15}{'gagnees':>10}{'ecart':>8}")
        print("-" * 74)
        for r in sorted(sel, key=lambda x: -x["auroc_pooled_abs"]):
            gag = f"{r['wins']}/{r['n']}"
            ecart = r["auroc_paired"] - r["auroc_pooled_abs"]
            print(f"{r['metric']:<18}{r['scale']:>9.2f}"
                  f"{r['auroc_pooled']:>14.3f}{r['auroc_paired']:>15.3f}"
                  f"{gag:>10}{ecart:>+8.3f}")
        best = max(sel, key=lambda x: x["auroc_pooled_abs"])
        nm = res["null_max"]
        pf = float((np.sum(nm >= best["auroc_pooled_abs"]) + 1) / (nm.size + 1))
        print()
        print(f"  maximum observe  : {best['auroc_pooled_abs']:.3f} "
              f"({best['metric']} @ s={best['scale']:g})")
        print(f"  nul intra-paires : moyenne {res['null_max_mean']:.3f}, "
              f"q95 {res['null_max_q95']:.3f}")
        print(f"  p FAMILIAL       : {pf:.4f}  "
              f"({args.n_perm} permutations)")
        if is_primary:
            print()
            if best["auroc_pooled_abs"] > res["null_max_q95"]:
                print(f"  -> CANDIDAT : {best['metric']} @ s={best['scale']:g}"
                      f"  (pooled {best['auroc_pooled_abs']:.3f}, "
                      f"apparie {best['auroc_paired']:.3f})")
            else:
                print("  -> AUCUN CANDIDAT : le maximum ne depasse pas le q95 "
                      "du nul.")
                print("     Condition d'arret pre-enregistree : pas de "
                      "lecture sur test.")
        print()

    print("ecart = apparie - pooled. C'est le gouffre que la standardisation")
    print("devait combler : s'il tombe vers zero, le detecteur n'a plus besoin")
    print("du controle apparie et devient utilisable sur un prompt isole.")
    print()

    if args.csv:
        os.makedirs(os.path.dirname(args.csv) or ".", exist_ok=True)
        with open(args.csv, "w", newline="", encoding="utf-8") as fh:
            w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)
        print(f"\n-> {args.csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
