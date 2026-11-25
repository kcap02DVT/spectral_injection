"""Figures et tests du spectre d'energie du flux residuel.

CE QUI EST TRACE. Pour chaque tete, ``mass_frac_k = m_k / sum_j m_j`` ou
``m_k = ||u_k^T X||^2`` : la part de l'energie du flux residuel portee par
le mode de frequence ``lambda_k``. La FRACTION, pas la masse brute : les
masses varient en ``c^2`` sous ``X -> cX``, donc un Delta sur ``m_k`` peut
etre porte par la seule norme du flux residuel. La fraction est invariante
par changement d'echelle. La masse brute est tracee a part, en annexe, pour
documenter la norme.

APPARIEMENT. Delta = C4 - C3_opi, calcule CELLULE PAR CELLULE puis agrege,
jamais l'inverse : pour la mediane les deux ordres ne coincident pas. La
comparaison par rang est definie parce que Delta N = 0 impose le meme
nombre de modes aux deux bras ; ``energy_spectrum.py`` le verifie par paire.

AUCUNE MOYENNE ENTRE TETES DANS LA MESURE. Les tests portent sur des
cellules (couche, tete) agregees par MEDIANE au niveau de la paire. Les
traits epais des figures sont une mediane d'affichage sur les tetes,
declaree comme telle ; les traits fins derriere sont les tetes elles-memes,
pour que la dispersion reste visible.

BANDES ET COUPURE DECLAREES D'AVANCE. Les bandes sont celles de
``spectrum_at_depth.py``, sur [0, 2] qui borne le spectre de L_sym. La
coupure haute frequence est ``lambda >= 1``, le milieu de [0, 2] : un point
fixe de l'operateur, pas un seuil ajuste. Aucune bande n'est choisie apres
avoir vu les resultats.

COULEURS. Deux series d'identite : bleu = benin (C3_opi), orange = adverse
(C4). Le panneau de Delta reutilise ces deux teintes comme poles divergents
-- orange au-dessus de zero signifie << plus de masse chez l'adverse >>,
bleu en dessous << plus de masse chez le benin >> -- avec un zero gris
neutre. La paire passe les six controles (Delta E CVD 24,7 ; normal 33,6).

    python tools/figures_energy_spectrum.py --root results/opi/energy_spectrum
"""

import argparse
import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from analyse_region_energy import dz, paired_test

ARM_BEN, ARM_ADV = "C3_opi", "C4"

#: Bandes de [0, 2], identiques a spectrum_at_depth.py. Declarees.
BANDS = ((0.0, 0.5), (0.5, 1.0), (1.0, 1.5), (1.5, 2.0 + 1e-9))
BAND_NAMES = ("band_0_05", "band_05_1", "band_1_15", "band_15_2")
#: Coupure haute frequence : le milieu de [0, 2]. Pas un seuil ajuste.
HIGH_CUT = 1.0

#: Points ou la fonction cumulative est EVALUEE, sur [0, 2].
#:
#: Ce n'est PAS une grille de valeurs propres imposee, et rien n'y est
#: interpole. ``F(lambda) = somme des m_k / somme des m_j sur les k tels que
#: lambda_k <= lambda`` est une fonction en escalier definie sur tout [0, 2]
#: par les lambda_k DE CE GRAPHE ; on la lit en 401 points. La valeur rendue
#: est exacte en chacun d'eux. C'est ce qui permet de superposer des
#: cellules de tailles differentes sans jamais apparier un mode a un mode.
EVAL_GRID = np.linspace(0.0, 2.0, 401)

# -- palette (surface claire #fcfcfb)
C_BEN = "#2a78d6"      # bleu, slot categoriel 1
C_ADV = "#eb6834"      # orange, slot categoriel 2
INK = "#0b0b0b"
INK2 = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
AXIS = "#c3c2b7"
SURF = "#fcfcfb"


def style():
    plt.rcParams.update({
        "figure.facecolor": SURF, "axes.facecolor": SURF,
        "savefig.facecolor": SURF,
        "font.family": "sans-serif",
        "font.sans-serif": ["Segoe UI", "DejaVu Sans", "sans-serif"],
        "axes.edgecolor": AXIS, "axes.linewidth": 0.8,
        "axes.labelcolor": INK2, "axes.titlecolor": INK,
        "axes.titlesize": 10.5, "axes.labelsize": 9,
        "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6,
        "xtick.color": MUTED, "ytick.color": MUTED,
        "xtick.labelsize": 8, "ytick.labelsize": 8,
        "legend.frameon": False, "legend.fontsize": 8.5,
        "axes.spines.top": False, "axes.spines.right": False,
    })


def band_table(M: pd.DataFrame) -> pd.DataFrame:
    """``mass_frac`` par bande et fraction haute frequence, par cellule.

    ``frac_high`` n'est pas une quantite independante : la coupure
    ``lambda >= 1`` tombe sur une frontiere de bande, donc elle vaut la
    somme des deux bandes hautes. Elle est calculee separement et l'egalite
    est VERIFIEE, ce qui controle du meme coup l'affectation des bandes.
    """
    lam = M["lambda"].to_numpy()
    bidx = np.full(lam.size, -1, dtype=np.int8)
    for b, (lo, hi) in enumerate(BANDS):
        bidx[(lam >= lo) & (lam < hi)] = b
    if (bidx < 0).any():
        bad = lam[bidx < 0]
        raise SystemExit(f"{bad.size} valeurs propres hors des bandes "
                         f"declarees [0, 2] : min {bad.min():.3e}, "
                         f"max {bad.max():.3e}")
    key = ["pair_id", "arm", "couche", "tete"]
    M = M.assign(_b=bidx, _w=lam * M["mass_frac"].to_numpy(),
                 _h=np.where(lam >= HIGH_CUT, M["mass_frac"].to_numpy(), 0.0))
    bt = (M.pivot_table(index=key, columns="_b", values="mass_frac",
                        aggfunc="sum").fillna(0.0)
          .rename(columns={b: BAND_NAMES[b] for b in range(len(BANDS))}))
    agg = M.groupby(key).agg(frac_high=("_h", "sum"), centroid=("_w", "sum"),
                             mass_total=("mass", "sum"))
    bt = bt.join(agg)
    gap = float((bt["frac_high"]
                 - bt[BAND_NAMES[2]] - bt[BAND_NAMES[3]]).abs().max())
    if gap > 1e-12:
        raise SystemExit(f"frac_high != somme des bandes hautes : {gap:.2e}")
    return bt.reset_index()


def paired_cells(bt: pd.DataFrame, quantities) -> pd.DataFrame:
    """Delta cellule par cellule, puis mediane par paire. Dans cet ordre."""
    piv = bt.pivot_table(index=["pair_id", "couche", "tete"], columns="arm",
                         values=list(quantities))
    out = {}
    for q in quantities:
        cell = piv[(q, ARM_ADV)] - piv[(q, ARM_BEN)]
        if cell.isna().any():
            raise SystemExit(f"appariement incomplet sur {q}")
        out[q] = cell
    D = pd.DataFrame(out)
    D.index.names = ["pair_id", "couche", "tete"]
    return D.reset_index()


def tests(Dcells: pd.DataFrame, quantities, layer: int) -> pd.DataFrame:
    """Wilcoxon apparie + dz sur les paires, a une couche donnee."""
    d = Dcells[Dcells.couche == layer]
    per_pair = d.groupby("pair_id")[list(quantities)].median()
    A = per_pair.to_numpy()
    p = paired_test(A)
    e = dz(A)
    agree = [(d[q] > 0).mean() for q in quantities]
    return pd.DataFrame({"quantite": list(quantities),
                         "delta_median": per_pair.median().to_numpy(),
                         "dz": e, "p_wilcoxon": p,
                         "n_paires": len(per_pair),
                         "cellules_positives": agree,
                         "n_cellules": len(d)})


def fig_pair(M: pd.DataFrame, pid: str, layer: int, out: str, tag: str):
    """Le spectre d'energie d'UNE paire, a la couche declaree."""
    g = M[(M.pair_id == pid) & (M.couche == layer)]
    heads = sorted(g.tete.unique())
    fig, ax = plt.subplots(1, 3, figsize=(13.2, 4.1))

    # -- (a) repartition de l'energie sur les frequences
    for arm, col, lab in ((ARM_BEN, C_BEN, "benin (C3_opi)"),
                          (ARM_ADV, C_ADV, "adverse (C4)")):
        a = g[g.arm == arm]
        for h in heads:
            s = a[a.tete == h].sort_values("rang")
            ax[0].plot(s["lambda"], s.mass_frac, color=col, lw=0.5, alpha=0.18)
        med = a.groupby("rang").agg(lam=("lambda", "median"),
                                    mf=("mass_frac", "median"))
        ax[0].plot(med.lam, med.mf, color=col, lw=2.0, label=lab, zorder=5)
    ax[0].set_yscale("log")
    ax[0].axvline(HIGH_CUT, color=AXIS, lw=0.8, ls=(0, (4, 3)), zorder=1)
    ax[0].annotate("$\\lambda = 1$", (HIGH_CUT, 1.0), xycoords=("data", "axes fraction"),
                   xytext=(3, -11), textcoords="offset points",
                   color=MUTED, fontsize=8)
    ax[0].set_xlabel("frequence de graphe  $\\lambda_k$   (spectre de $L_{sym}$)")
    ax[0].set_ylabel("part de l'energie du mode   $m_k / \\sum_j m_j$")
    ax[0].set_title("(a) repartition de l'energie du flux residuel\n"
                    f"{len(heads)} tetes, couche {layer}", loc="left")
    ax[0].legend(loc="lower left")

    # -- (b) masse cumulee : lecture robuste aux quasi-degenerescences
    for arm, col, lab in ((ARM_BEN, C_BEN, "benin (C3_opi)"),
                          (ARM_ADV, C_ADV, "adverse (C4)")):
        a = g[g.arm == arm]
        for h in heads:
            s = a[a.tete == h].sort_values("rang")
            ax[1].plot(s["lambda"], s.mass_frac.cumsum(), color=col, lw=0.5,
                       alpha=0.18)
        cum = (a.sort_values("rang").groupby("tete")["mass_frac"].cumsum()
               .rename("c"))
        cc = a.assign(c=cum).groupby("rang").agg(lam=("lambda", "median"),
                                                 c=("c", "median"))
        ax[1].plot(cc.lam, cc.c, color=col, lw=2.0, label=lab, zorder=5)
    ax[1].axvline(HIGH_CUT, color=AXIS, lw=0.8, ls=(0, (4, 3)), zorder=1)
    ax[1].set_xlabel("frequence de graphe  $\\lambda_k$")
    ax[1].set_ylabel("energie cumulee  $\\sum_{j \\leq k} m_j / \\sum_j m_j$")
    ax[1].set_title("(b) energie cumulee\n"
                    "insensible aux quasi-degenerescences", loc="left")
    ax[1].legend(loc="upper left")

    # -- (c) Delta apparie de la masse cumulee, par rang
    piv = g.pivot_table(index=["tete", "rang"], columns="arm",
                        values="mass_frac")
    cum = piv.groupby("tete").cumsum()
    d = (cum[ARM_ADV] - cum[ARM_BEN]).rename("d").reset_index()
    for h in heads:
        s = d[d.tete == h]
        ax[2].plot(s.rang, s.d, color=MUTED, lw=0.5, alpha=0.30)
    med = d.groupby("rang")["d"].median()
    ax[2].axhline(0.0, color=AXIS, lw=1.0, zorder=2)
    ax[2].fill_between(med.index, 0, med.values, where=med.values >= 0,
                       color=C_ADV, alpha=0.22, lw=0, zorder=3)
    ax[2].fill_between(med.index, 0, med.values, where=med.values < 0,
                       color=C_BEN, alpha=0.22, lw=0, zorder=3)
    ax[2].plot(med.index, med.values, color=INK, lw=2.0, zorder=5)
    ax[2].set_xlabel("rang du mode  $k$   (0 = plus basse frequence)")
    ax[2].set_ylabel("$\\Delta$ energie cumulee   (C4 $-$ C3_opi)")
    ax[2].set_title("(c) $\\Delta$ apparie de l'energie cumulee\n"
                    "orange = plus de masse chez l'adverse", loc="left")
    # Bornes robustes : quelques tetes atteignent des amplitudes dix fois
    # superieures et ecraseraient la mediane, qui est la lecture.
    span = float(np.nanpercentile(np.abs(d.d.to_numpy()), 97))
    span = max(span, float(np.abs(med.to_numpy()).max()), 1e-12)
    ax[2].set_ylim(-1.5 * span, 1.5 * span)
    n_out = int((d.groupby("tete")["d"].apply(lambda s: s.abs().max())
                 > 1.5 * span).sum())
    if n_out:
        ax[2].annotate(f"{n_out} tetes sortent du cadre", (0.03, 0.05),
                       xycoords="axes fraction", color=MUTED, fontsize=7.5)

    fig.suptitle(f"Spectre d'energie du flux residuel sur le graphe "
                 f"d'attention  —  {tag}  —  paire {pid}",
                 x=0.012, ha="left", fontsize=12, color=INK)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    p = os.path.join(out, f"spectre_energie__{tag}__paire.png")
    fig.savefig(p, dpi=170)
    plt.close(fig)
    return p


def fig_paired(M: pd.DataFrame, Dcells: pd.DataFrame, layer: int,
               out: str, tag: str, n_pairs: int):
    """L'agregat apparie : toutes les paires, toutes les cellules."""
    fig, ax = plt.subplots(1, 3, figsize=(13.2, 4.1))

    # -- (a) Delta median de mass_frac par rang, toutes paires x cellules
    piv = M[M.couche == layer].pivot_table(
        index=["pair_id", "tete", "rang"], columns="arm", values="mass_frac")
    d = (piv[ARM_ADV] - piv[ARM_BEN]).rename("d").reset_index()
    q = d.groupby("rang")["d"].quantile([0.25, 0.5, 0.75]).unstack()
    ax[0].axhline(0.0, color=AXIS, lw=1.0, zorder=2)
    ax[0].fill_between(q.index, q[0.25], q[0.75], color=MUTED, alpha=0.18,
                       lw=0, zorder=3, label="interquartile des cellules")
    m = q[0.5]
    ax[0].fill_between(q.index, 0, m.values, where=m.values >= 0, color=C_ADV,
                       alpha=0.30, lw=0, zorder=4)
    ax[0].fill_between(q.index, 0, m.values, where=m.values < 0, color=C_BEN,
                       alpha=0.30, lw=0, zorder=4)
    ax[0].plot(q.index, m.values, color=INK, lw=1.8, zorder=5,
               label="mediane des cellules")
    ax[0].set_xlabel("rang du mode  $k$")
    ax[0].set_ylabel("$\\Delta$ part d'energie du mode")
    ax[0].set_title(f"(a) $\\Delta$ apparie par mode\n"
                    f"{n_pairs} paires, couche {layer}", loc="left")
    ax[0].legend(loc="upper left")

    # -- (b) Delta par bande declaree, barres divergentes
    dd = Dcells[Dcells.couche == layer]
    names = list(BAND_NAMES)
    med = [dd[n].median() for n in names]
    pos = [float((dd[n] > 0).mean()) for n in names]
    cols = [C_ADV if v >= 0 else C_BEN for v in med]
    y = np.arange(len(names))
    ax[1].axvline(0.0, color=AXIS, lw=1.0, zorder=2)
    ax[1].barh(y, med, color=cols, height=0.62, zorder=3)
    ax[1].set_yticks(y)
    ax[1].set_yticklabels(["$\\lambda \\in [0{,}0 ; 0{,}5[$",
                           "$\\lambda \\in [0{,}5 ; 1{,}0[$",
                           "$\\lambda \\in [1{,}0 ; 1{,}5[$",
                           "$\\lambda \\in [1{,}5 ; 2{,}0]$"])
    ax[1].invert_yaxis()
    ax[1].grid(axis="y", visible=False)
    span = max(abs(min(med)), abs(max(med))) or 1.0
    # Toutes les etiquettes a DROITE de l'extremite de la barre : a gauche
    # elles chevauchaient les graduations de l'axe des ordonnees.
    for i, (v, pp) in enumerate(zip(med, pos)):
        txt = f"{v:+.1e} · {100 * pp:.0f} % des cellules"
        if abs(v) < 1e-18:
            txt = f"mediane nulle · {100 * pp:.0f} % des cellules"
        ax[1].annotate(txt, (max(v, 0.0), i), xytext=(7, 0),
                       textcoords="offset points", va="center", ha="left",
                       color=INK2, fontsize=8)
    ax[1].set_xlim(-1.6 * span, 3.6 * span)
    ax[1].set_xlabel("$\\Delta$ part d'energie dans la bande  (C4 $-$ C3_opi)")
    ax[1].set_title("(b) $\\Delta$ par bande de frequence\n"
                    "bandes declarees, non ajustees", loc="left")

    # -- (c) profil de profondeur de la fraction haute frequence
    prof = (Dcells.groupby("couche")["frac_high"]
            .quantile([0.25, 0.5, 0.75]).unstack())
    agree = Dcells.groupby("couche")["frac_high"].apply(lambda s: (s > 0).mean())
    ax[2].axhline(0.0, color=AXIS, lw=1.0, zorder=2)
    ax[2].fill_between(prof.index, prof[0.25], prof[0.75], color=MUTED,
                       alpha=0.18, lw=0, zorder=3)
    ax[2].plot(prof.index, prof[0.5], color=INK, lw=2.0, zorder=5,
               marker="o", ms=4)
    ax[2].axvline(layer, color=AXIS, lw=0.8, ls=(0, (4, 3)), zorder=1)
    ax[2].annotate(f"couche declaree {layer}", (layer, 1.0),
                   xycoords=("data", "axes fraction"), xytext=(4, -11),
                   textcoords="offset points", color=MUTED, fontsize=8)
    ax[2].set_xlabel("couche")
    ax[2].set_ylabel("$\\Delta$ fraction d'energie a  $\\lambda \\geq 1$")
    ax[2].set_title("(c) profil de profondeur\n"
                    "mediane et interquartile des cellules", loc="left")
    # Part de cellules de signe positif, annotee directement au-dessus des
    # points : un second axe des abscisses se lisait comme une echelle.
    for c in list(prof.index)[::3]:
        ax[2].annotate(f"{100 * agree.loc[c]:.0f} %", (c, prof.loc[c, 0.5]),
                       xytext=(0, 9), textcoords="offset points", ha="center",
                       color=MUTED, fontsize=7.5)
    ax[2].annotate("part des cellules de signe positif", (0.03, 0.05),
                   xycoords="axes fraction", color=MUTED, fontsize=7.5)
    ax[2].set_xlim(-0.9, len(prof) - 0.1)

    fig.suptitle(f"$\\Delta$ apparie du spectre d'energie  —  {tag}  —  "
                 f"{n_pairs} paires, C4 $-$ C3_opi",
                 x=0.012, ha="left", fontsize=12, color=INK)
    fig.tight_layout(rect=(0, 0, 1, 0.91))
    p = os.path.join(out, f"spectre_energie__{tag}__apparie.png")
    fig.savefig(p, dpi=170)
    plt.close(fig)
    return p


def lambda_divergence(M: pd.DataFrame, layer: int) -> pd.DataFrame:
    """De combien les deux grilles de frequences diffErent, par rang.

    Mesure a titre de DIAGNOSTIC seulement : elle quantifie l'erreur qu'on
    commettrait en lisant les deux spectres sur un axe commun. Elle ne sert
    a aucune comparaison, et les figures a deux panneaux ne s'en servent
    pas -- chaque panneau garde ses propres lambda.
    """
    piv = M[M.couche == layer].pivot_table(
        index=["pair_id", "tete", "rang"], columns="arm", values="lambda")
    d = (piv[ARM_ADV] - piv[ARM_BEN]).abs()
    return pd.DataFrame({"ecart_median": [float(d.median())],
                         "ecart_q99": [float(d.quantile(0.99))],
                         "ecart_max": [float(d.max())]})


def _arm_descriptors(a: pd.DataFrame) -> dict:
    """Descripteurs du spectre d'UN bras, calcules sur SES propres lambda."""
    lam = a["lambda"].to_numpy()
    mf = a["mass_frac"].to_numpy()
    per_head = a.sort_values("rang").groupby("tete")["lambda"]
    return {"lambda_2": float(per_head.nth(1).median()),
            "lambda_max": float(per_head.max().median()),
            "frac_high": float(mf[lam >= HIGH_CUT].sum() / mf.sum()),
            "centroid": float((lam * mf).sum() / mf.sum())}


def fig_own(M: pd.DataFrame, pid: str, layer: int, out: str, tag: str):
    """Les deux spectres d'UNE paire, chacun sur ses propres frequences.

    DEUX PANNEAUX SEPARES, volontairement. Les deux bras ont le meme nombre
    de modes (Delta N = 0) mais PAS les memes valeurs propres : superposer
    les deux courbes sur un axe unique obligerait a choisir l'un des deux
    jeux de lambda, ou a apparier par rang, ce qui fabrique une
    correspondance que l'algebre ne donne pas. Chaque panneau porte donc le
    spectre de son graphe, lu sur les lambda de ce graphe.

    ECHELLES PARTAGEES. Abscisse [0, 2] -- la borne du spectre de L_sym,
    valable pour les deux graphes sans hypothese -- et meme ordonnee
    logarithmique, pour que la comparaison visuelle soit honnete.
    """
    g = M[(M.pair_id == pid) & (M.couche == layer)]
    heads = sorted(g.tete.unique())
    T = int(g.rang.max()) + 1
    fig, ax = plt.subplots(1, 2, figsize=(12.4, 4.6), sharey=True)

    specs = ((ARM_BEN, C_BEN, "benin  (C3_opi)"),
             (ARM_ADV, C_ADV, "adverse  (C4)"))
    for k, (arm, col, lab) in enumerate(specs):
        a = g[g.arm == arm]
        for h in heads:
            s = a[a.tete == h].sort_values("rang")
            ax[k].plot(s["lambda"], s.mass_frac, color=col, lw=0.5, alpha=0.20)
        med = a.groupby("rang").agg(lam=("lambda", "median"),
                                    mf=("mass_frac", "median"))
        ax[k].plot(med.lam, med.mf, color=col, lw=2.0, zorder=5)
        d = _arm_descriptors(a)
        ax[k].axvline(HIGH_CUT, color=AXIS, lw=0.8, ls=(0, (4, 3)), zorder=1)
        ax[k].set_xlim(-0.03, 2.03)
        ax[k].set_xticks([0.0, 0.5, 1.0, 1.5, 2.0])
        ax[k].set_xlabel("frequences du graphe de CE bras   "
                         "$\\lambda_k$  (spectre de $L_{sym}$)")
        # Un seul jeu de donnees par panneau : le titre le nomme, pas de
        # boite de legende.
        ax[k].set_title(f"({'ab'[k]}) {lab}\n"
                        f"{len(heads)} tetes, couche {layer}, "
                        f"T = {T} modes par tete", loc="left")
        txt = (f"$\\lambda_2$ = {d['lambda_2']:.4f}\n"
               f"$\\lambda_{{max}}$ = {d['lambda_max']:.4f}\n"
               f"centroide = {d['centroid']:.4f}\n"
               f"energie a $\\lambda \\geq 1$ : "
               f"{100 * d['frac_high']:.1f} %")
        ax[k].annotate(txt, (0.985, 0.03), xycoords="axes fraction",
                       ha="right", va="bottom", color=INK2, fontsize=8.5,
                       linespacing=1.5)
        ax[k].annotate("$\\lambda = 1$", (HIGH_CUT, 1.0),
                       xycoords=("data", "axes fraction"), xytext=(3, -11),
                       textcoords="offset points", color=MUTED, fontsize=8)
    ax[0].set_yscale("log")
    ax[0].set_ylabel("part de l'energie du mode   $m_k / \\sum_j m_j$")

    fig.suptitle(f"Spectre d'energie du flux residuel, chaque signal sur la "
                 f"base propre de SON graphe  —  {tag}  —  paire {pid}",
                 x=0.012, ha="left", fontsize=12, color=INK)
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    p = os.path.join(out, f"spectre_energie__{tag}__propre_paire.png")
    fig.savefig(p, dpi=170)
    plt.close(fig)
    return p


def cumulative_energy(M: pd.DataFrame, layer: int):
    """``F(lambda)`` exacte par cellule, lue sur ``EVAL_GRID``.

    Pour chaque cellule (bras, paire, tete), ``F`` est la fonction en
    escalier qui monte de ``m_k / sum_j m_j`` a chaque ``lambda_k`` DE CE
    GRAPHE. On la lit aux points de la grille ; aucune valeur propre n'est
    deplacee, aucune n'est appariee a celle de l'autre bras, et des
    cellules de tailles differentes deviennent comparables parce que ``F``
    est definie sur tout [0, 2] dans les deux cas.
    """
    d = M[M.couche == layer]
    out = {}
    for arm in (ARM_BEN, ARM_ADV):
        rows = []
        for _, c in d[d.arm == arm].groupby(["pair_id", "tete"], sort=True):
            c = c.sort_values("lambda")
            cum = np.concatenate(([0.0], np.cumsum(c.mass_frac.to_numpy())))
            idx = np.searchsorted(c["lambda"].to_numpy(), EVAL_GRID,
                                  side="right")
            rows.append(cum[idx])
        A = np.asarray(rows)
        if abs(A[:, -1].max() - 1.0) > 1e-9 or abs(A[:, -1].min() - 1.0) > 1e-9:
            raise SystemExit("F(2) != 1 : des valeurs propres sortent de "
                             "[0, 2], le spectre n'est pas celui de L_sym")
        out[arm] = A
    return out


def fig_own_pairs(M: pd.DataFrame, layer: int, out: str, tag: str,
                  n_pairs: int):
    """Les deux spectres de toutes les paires, chacun sur ses frequences.

    Les paires ont des T differents, donc aucun rang n'est comparable d'une
    paire a l'autre et aucun intervalle de frequence ne contient le meme
    nombre de modes d'une cellule a l'autre. L'objet agregeable est la
    fonction CUMULATIVE : elle est definie sur tout [0, 2] quelle que soit
    la taille du graphe, et sa mediane entre cellules reste une fonction
    croissante de 0 a 1, donc lisible.

    Un histogramme de densite serait trompeur ici : le pic de haute
    frequence est porte par le dernier mode, dont le ``lambda`` varie d'une
    cellule a l'autre ; dans un intervalle fixe la mediane entre cellules
    vaut alors zero et le pic disparait de la figure alors qu'il porte la
    moitie de l'energie.
    """
    C = cumulative_energy(M, layer)
    fig, ax = plt.subplots(1, 2, figsize=(12.4, 4.6), sharey=True)
    specs = ((ARM_BEN, C_BEN, "benin  (C3_opi)"),
             (ARM_ADV, C_ADV, "adverse  (C4)"))
    for k, (arm, col, lab) in enumerate(specs):
        A = C[arm]
        q = np.percentile(A, [25, 50, 75], axis=0)
        ax[k].fill_between(EVAL_GRID, q[0], q[2], color=col, alpha=0.22,
                           lw=0, zorder=3)
        ax[k].plot(EVAL_GRID, q[1], color=col, lw=2.0, zorder=5)
        ax[k].axvline(HIGH_CUT, color=AXIS, lw=0.8, ls=(0, (4, 3)), zorder=1)
        ax[k].set_xlim(-0.03, 2.03)
        ax[k].set_xticks([0.0, 0.5, 1.0, 1.5, 2.0])
        ax[k].set_ylim(0.0, 1.03)
        ax[k].set_xlabel("frequences du graphe de CE bras   $\\lambda_k$")
        ax[k].set_title(f"({'ab'[k]}) {lab}\n"
                        f"{n_pairs} paires x {A.shape[0] // n_pairs} tetes, "
                        f"couche {layer}", loc="left")
        j = int(np.searchsorted(EVAL_GRID, HIGH_CUT, side="right")) - 1
        lo = float(np.median(A[:, j]))
        ax[k].annotate(f"energie a $\\lambda < 1$ : {100 * lo:.1f} %\n"
                       f"energie a $\\lambda \\geq 1$ : {100 * (1 - lo):.1f} %\n"
                       f"mediane et interquartile des cellules",
                       (0.985, 0.06), xycoords="axes fraction", ha="right",
                       va="bottom", color=INK2, fontsize=8.5, linespacing=1.5)
        ax[k].annotate("$\\lambda = 1$", (HIGH_CUT, 1.0),
                       xycoords=("data", "axes fraction"), xytext=(3, -11),
                       textcoords="offset points", color=MUTED, fontsize=8)
    ax[0].set_ylabel("energie cumulee   "
                     "$F(\\lambda) = \\sum_{\\lambda_k \\leq \\lambda} "
                     "m_k / \\sum_j m_j$")

    fig.suptitle(f"Spectres d'energie apparies, chacun sur la base propre de "
                 f"son graphe  —  {tag}  —  {n_pairs} paires",
                 x=0.012, ha="left", fontsize=12, color=INK)
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    p = os.path.join(out, f"spectre_energie__{tag}__propre_apparie.png")
    fig.savefig(p, dpi=170)
    plt.close(fig)
    return p


def cumulative_all(M: pd.DataFrame):
    """``F(lambda)`` exacte pour TOUTES les couches, par cellule.

    Meme objet que ``cumulative_energy``, calcule en un seul passage pour
    les 16 couches. Rend ``{(bras, couche): tableau [n_cellules, 401]}``.
    """
    d = M.sort_values(["arm", "couche", "pair_id", "tete", "lambda"],
                      kind="mergesort")
    lam = d["lambda"].to_numpy()
    mf = d["mass_frac"].to_numpy()
    key = pd.MultiIndex.from_frame(d[["arm", "couche", "pair_id", "tete"]])
    codes, uniq = pd.factorize(key)
    start = np.flatnonzero(np.r_[True, codes[1:] != codes[:-1]])
    end = np.r_[start[1:], codes.size]
    out = {}
    for s, e in zip(start, end):
        arm, couche = uniq[codes[s]][0], uniq[codes[s]][1]
        cum = np.r_[0.0, np.cumsum(mf[s:e])]
        out.setdefault((arm, int(couche)), []).append(
            cum[np.searchsorted(lam[s:e], EVAL_GRID, side="right")])
    out = {k: np.asarray(v) for k, v in out.items()}
    worst = max(abs(A[:, -1] - 1.0).max() for A in out.values())
    if worst > 1e-9:
        raise SystemExit(f"F(2) != 1 : ecart {worst:.2e}")
    return out


def _ramp(c_lo: str, c_hi: str, n: int):
    """Rampe sequentielle d'une seule teinte, clair -> fonce, n pas."""
    from matplotlib.colors import LinearSegmentedColormap
    cm = LinearSegmentedColormap.from_list("r", [c_lo, c_hi])
    return cm, [cm(i / max(n - 1, 1)) for i in range(n)]


def fig_own_depth(M: pd.DataFrame, out: str, tag: str, n_pairs: int,
                  primary: int):
    """Les spectres de TOUTES les couches, chaque bras sur son propre panneau.

    Une courbe par couche, et la profondeur est encodee par une rampe
    SEQUENTIELLE d'une seule teinte -- clair = entree, fonce = sortie --
    parce que la couche est une grandeur ordonnee, pas une identite. La
    teinte de la rampe est celle du bras, de sorte que l'identite reste
    portee par le panneau et la profondeur par la luminosite.

    L'objet trace est la cumulative, pour la meme raison qu'en figure
    appariee : les 16 paires n'ont pas le meme T, donc l'axe des rangs n'est
    pas agregeable, tandis que F est definie sur tout [0, 2] partout.
    """
    C = cumulative_all(M)
    layers = sorted({l for _, l in C})
    fig, ax = plt.subplots(1, 2, figsize=(12.8, 4.9), sharey=True)
    specs = ((ARM_BEN, "#9dc2ed", "#0d3663", "benin  (C3_opi)"),
             (ARM_ADV, "#f5b998", "#7d2c0b", "adverse  (C4)"))
    for k, (arm, lo, hi, lab) in enumerate(specs):
        cm, cols = _ramp(lo, hi, len(layers))
        for i, l in enumerate(layers):
            A = C[(arm, l)]
            ax[k].plot(EVAL_GRID, np.median(A, axis=0), color=cols[i],
                       lw=1.6, zorder=3 + i)
        j = int(np.searchsorted(EVAL_GRID, HIGH_CUT, side="right")) - 1
        for l, dy in ((layers[0], 7), (layers[-1], -13)):
            v = float(np.median(C[(arm, l)][:, j]))
            ax[k].annotate(f"couche {l}", (HIGH_CUT, v), xytext=(6, dy),
                           textcoords="offset points", fontsize=8,
                           color=INK2, va="center")
        ax[k].axvline(HIGH_CUT, color=AXIS, lw=0.8, ls=(0, (4, 3)), zorder=1)
        ax[k].set_xlim(-0.03, 2.03)
        ax[k].set_xticks([0.0, 0.5, 1.0, 1.5, 2.0])
        ax[k].set_ylim(0.0, 1.03)
        ax[k].set_xlabel("frequences du graphe de CE bras   $\\lambda_k$")
        ax[k].set_title(f"({'ab'[k]}) {lab}\n{len(layers)} couches, "
                        f"{n_pairs} paires x {C[(arm, layers[0])].shape[0] // n_pairs}"
                        f" tetes par couche", loc="left")
        sm = plt.cm.ScalarMappable(cmap=cm, norm=plt.Normalize(
            layers[0], layers[-1]))
        cb = fig.colorbar(sm, ax=ax[k], pad=0.015, fraction=0.045)
        cb.set_label("couche  (clair = entree, fonce = sortie)", fontsize=8,
                     color=INK2)
        cb.outline.set_visible(False)
        cb.ax.tick_params(labelsize=7.5, color=AXIS)
    ax[0].set_ylabel("energie cumulee   "
                     "$F(\\lambda) = \\sum_{\\lambda_k \\leq \\lambda} "
                     "m_k / \\sum_j m_j$")

    fig.suptitle(f"Spectre d'energie sur toute la profondeur, chaque signal "
                 f"sur la base propre de son graphe  —  {tag}  —  "
                 f"{n_pairs} paires  (couche declaree : {primary})",
                 x=0.012, ha="left", fontsize=12, color=INK)
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    p = os.path.join(out, f"spectre_energie__{tag}__propre_couches.png")
    fig.savefig(p, dpi=170)
    plt.close(fig)
    return p, C


def fig_own_grid(M: pd.DataFrame, pid: str, arm: str, out: str, tag: str):
    """Une couche par panneau : la DENSITE spectrale d'une paire, 32 tetes.

    La cumulative repond a << ou est l'energie >>, pas a << quelle est la
    forme du spectre >>. Ce quadrillage rend la densite, tete par tete,
    couche par couche, pour une seule paire -- la densite n'est agregeable
    ni entre paires de T differents ni par intervalle fixe.
    """
    g = M[(M.pair_id == pid) & (M.arm == arm)]
    layers = sorted(g.couche.unique())
    col = C_BEN if arm == ARM_BEN else C_ADV
    lab = "benin  (C3_opi)" if arm == ARM_BEN else "adverse  (C4)"
    nc = 4
    nr = int(np.ceil(len(layers) / nc))
    fig, axs = plt.subplots(nr, nc, figsize=(13.0, 2.75 * nr),
                            sharex=True, sharey=True)
    axs = np.atleast_1d(axs).ravel()
    for i, l in enumerate(layers):
        a = g[g.couche == l]
        for h in sorted(a.tete.unique()):
            s = a[a.tete == h].sort_values("rang")
            axs[i].plot(s["lambda"], s.mass_frac, color=col, lw=0.45,
                        alpha=0.20)
        med = a.groupby("rang").agg(lam=("lambda", "median"),
                                    mf=("mass_frac", "median"))
        axs[i].plot(med.lam, med.mf, color=col, lw=1.6, zorder=5)
        axs[i].axvline(HIGH_CUT, color=AXIS, lw=0.7, ls=(0, (4, 3)), zorder=1)
        axs[i].set_yscale("log")
        axs[i].set_xlim(-0.03, 2.03)
        axs[i].set_xticks([0.0, 0.5, 1.0, 1.5, 2.0])
        lam = a["lambda"].to_numpy()
        mf = a.mass_frac.to_numpy()
        axs[i].set_title(f"couche {l}   "
                         f"$\\lambda \\geq 1$ : "
                         f"{100 * mf[lam >= HIGH_CUT].sum() / mf.sum():.0f} %",
                         loc="left", fontsize=9.5)
    for j in range(len(layers), axs.size):
        axs[j].set_visible(False)
    for j in range(axs.size - nc, axs.size):
        if axs[j].get_visible():
            axs[j].set_xlabel("$\\lambda_k$")
    for j in range(0, axs.size, nc):
        axs[j].set_ylabel("$m_k / \\sum_j m_j$")

    fig.suptitle(f"Densite spectrale couche par couche  —  {lab}  —  {tag}  "
                 f"—  paire {pid}", x=0.012, ha="left", fontsize=12, color=INK)
    fig.tight_layout(rect=(0, 0, 1, 1 - 0.055 * 4 / nr))
    sfx = "benin" if arm == ARM_BEN else "adverse"
    p = os.path.join(out, f"spectre_energie__{tag}__propre_grille_{sfx}.png")
    fig.savefig(p, dpi=150)
    plt.close(fig)
    return p


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=os.path.join("results", "opi",
                                                   "energy_spectrum"))
    ap.add_argument("--tag", default=None)
    args = ap.parse_args()
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass
    style()

    tags = ([args.tag] if args.tag else
            [t for t in sorted(os.listdir(args.root))
             if os.path.exists(os.path.join(args.root, t,
                                            "masses_long.parquet"))])
    for tag in tags:
        d = os.path.join(args.root, tag)
        with open(os.path.join(d, "provenance.json"), encoding="utf-8") as fh:
            prov = json.load(fh)
        layer = int(prov["primary_layer"])
        M = pd.read_parquet(os.path.join(d, "masses_long.parquet"))
        pairs = sorted(M.pair_id.unique())
        print("=" * 78)
        print(f"{tag}  —  spectral-trust {prov['spectral_trust']}, famille "
              f"{prov['family']}, signal {prov['signal']}")
        print(f"{len(pairs)} paires, {prov['n_layers']} couches x "
              f"{prov['n_heads']} tetes, couche declaree {layer} "
              f"({prov['primary_frac']:.2f} de la profondeur)")
        chk = pd.read_csv(os.path.join(d, "integrity.csv"))
        print(f"Parseval pire cas {chk.parseval_max.max():.2e}, "
              f"orthonormalite {chk.ortho_max.max():.2e}, "
              f"T de {chk.n_tokens.min()} a {chk.n_tokens.max()}")

        bt = band_table(M)
        QUANT = list(BAND_NAMES) + ["frac_high", "centroid", "mass_total"]
        Dcells = paired_cells(bt, QUANT)
        T = tests(Dcells, QUANT, layer)
        T.to_csv(os.path.join(d, "tests_primary_layer.csv"), index=False)
        Dcells.to_parquet(os.path.join(d, "delta_cells.parquet"), index=False)

        print(f"\nCouche {layer}. Delta = C4 - C3_opi, par cellule puis "
              f"mediane par paire. Wilcoxon apparie bilateral.")
        print(f"{'quantite':12s} {'Delta median':>14s} {'dz':>7s} "
              f"{'p':>9s} {'cellules +':>11s}")
        for _, r in T.iterrows():
            print(f"{r.quantite:12s} {r.delta_median:+14.3e} {r.dz:+7.2f} "
                  f"{r.p_wilcoxon:9.2e} {100 * r.cellules_positives:10.0f} %")

        prof = (Dcells.groupby("couche")["frac_high"]
                .agg(median="median",
                     pos=lambda s: (s > 0).mean()).reset_index())
        prof.to_csv(os.path.join(d, "depth_profile_frac_high.csv"), index=False)
        print(f"\nProfil de profondeur de frac_high (lambda >= {HIGH_CUT}) :")
        for _, r in prof.iterrows():
            bar = "#" * int(round(40 * abs(r["median"]) /
                                  max(prof["median"].abs().max(), 1e-12)))
            print(f"  couche {int(r.couche):2d}  {r['median']:+.3e}  "
                  f"{100 * r['pos']:3.0f} %  {bar}")

        # -- Lecture SEPAREE par bras, sur les lambda de chaque graphe.
        # Aucune soustraction : on decrit deux distributions, on ne les
        # apparie pas mode a mode.
        print(f"\nLecture separee par bras, couche {layer}. Chaque bras sur "
              f"ses propres valeurs propres, mediane des cellules.")
        per_arm = (bt[bt.couche == layer]
                   .groupby("arm")[list(BAND_NAMES) + ["frac_high", "centroid"]]
                   .median())
        per_arm.to_csv(os.path.join(d, "bands_par_bras.csv"))
        print(f"{'bras':10s}" + "".join(f"{n:>13s}" for n in BAND_NAMES)
              + f"{'frac_high':>13s}{'centroide':>12s}")
        for arm in (ARM_BEN, ARM_ADV):
            r = per_arm.loc[arm]
            print(f"{arm:10s}" + "".join(f"{r[n]:13.5f}" for n in BAND_NAMES)
                  + f"{r.frac_high:13.5f}{r.centroid:12.5f}")

        LD = lambda_divergence(M, layer)
        LD.to_csv(os.path.join(d, "lambda_divergence.csv"), index=False)
        print(f"\nEcart |lambda_k(C4) - lambda_k(C3_opi)| au MEME rang : "
              f"median {LD.ecart_median[0]:.3e}, q99 {LD.ecart_q99[0]:.3e}, "
              f"max {LD.ecart_max[0]:.3e}")
        print("  diagnostic seul : mesure l'erreur qu'on ferait en lisant les "
              "deux spectres sur un axe commun. Les figures a deux panneaux "
              "ne l'utilisent pas.")

        # -- Toutes les couches, lecture SEPAREE par bras, sans Delta.
        p_depth, C = fig_own_depth(M, d, tag, len(pairs), layer)
        jc = int(np.searchsorted(EVAL_GRID, HIGH_CUT, side="right")) - 1
        prof_arm = pd.DataFrame(
            [{"couche": l, "arm": arm,
              "frac_high_mediane": float(1.0 - np.median(C[(arm, l)][:, jc])),
              "n_cellules": int(C[(arm, l)].shape[0])}
             for l in sorted({x for _, x in C}) for arm in (ARM_BEN, ARM_ADV)])
        prof_arm.to_csv(os.path.join(d, "frac_high_par_bras_profondeur.csv"),
                        index=False)
        print(f"\nFraction d'energie a lambda >= {HIGH_CUT}, par bras et par "
              f"couche. Mediane des cellules, chaque bras sur ses propres "
              f"valeurs propres. Aucune soustraction.")
        print(f"{'couche':>7s} {ARM_BEN:>12s} {ARM_ADV:>12s}")
        for l in sorted({x for _, x in C}):
            b = prof_arm[(prof_arm.couche == l)].set_index("arm")
            mark = "  <== declaree" if l == layer else ""
            print(f"{l:7d} {b.loc[ARM_BEN, 'frac_high_mediane']:12.5f} "
                  f"{b.loc[ARM_ADV, 'frac_high_mediane']:12.5f}{mark}")

        figs = [p_depth,
                fig_own_grid(M, pairs[0], ARM_BEN, d, tag),
                fig_own_grid(M, pairs[0], ARM_ADV, d, tag),
                fig_own(M, pairs[0], layer, d, tag),
                fig_own_pairs(M, layer, d, tag, len(pairs)),
                fig_pair(M, pairs[0], layer, d, tag),
                fig_paired(M, Dcells, layer, d, tag, len(pairs))]
        print("\nfigures :")
        for p in figs:
            print(f"  {p}")
        print(f"tables  : {d}/tests_primary_layer.csv, "
              f"depth_profile_frac_high.csv, delta_cells.parquet")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
