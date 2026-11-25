"""Figures du spectre d'energie PAR REGION.

Meme grammaire que figures_energy_spectrum.py : densite en echelle log,
echelle frequentielle [0, 2] imposee partout, aucune valeur propre commune
entre bras, 32 courbes de tetes en transparence plus la mediane en trait
plein, trait tirete a lambda = 1.

QUATRE FIGURES.

1. ``regions_couche<l>``  -- 2 bras x 6 regions, a la couche declaree.
   Vue "propre" : chaque region sur les vecteurs propres de SON sous-graphe
   induit. C'est la figure demandee : les regions au lieu du graphe entier.

2. ``grille_benin`` / ``grille_adverse`` -- une couche par panneau, pour la
   REGION DECLAREE (``slot``). C'est l'analogue direct de
   ``spectre_energie__*__propre_grille_adverse.png``.

3. ``cumulative`` -- la fonction cumulee exacte F_R(lambda), une region par
   panneau, les deux bras superposes. OBLIGATOIRE pour comparer des
   regions entre elles : elles n'ont pas le meme nombre de tokens, donc pas
   le meme nombre de modes, et deux densites de supports differents ne se
   comparent pas. La cumulee est definie sur tout [0, 2] quel que soit n_R.

4. ``partagee_couche<l>`` -- vue complementaire : le graphe reste ENTIER,
   seul le signal est restreint a la region. Les six distributions vivent
   alors sur les memes frequences. Repond a << a quelle frequence du graphe
   complet se loge l'energie de chaque region >>.
"""

import argparse
import json
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt                                 # noqa: E402

from analyse_region_spectrum import _cles, _kolmogorov          # noqa: E402

ARM_BEN, ARM_ADV = "C3_opi", "C4"
HIGH_CUT = 1.0
EVAL_GRID = np.linspace(0.0, 2.0, 401)

C_BEN = "#2a78d6"
C_ADV = "#eb6834"
INK, INK2, MUTED = "#0b0b0b", "#52514e", "#898781"
GRID, AXIS, SURF = "#e1e0d9", "#c3c2b7", "#fcfcfb"

NOM_BRAS = {ARM_BEN: "benin  (C3_opi)", ARM_ADV: "adverse  (C4)"}
COUL = {ARM_BEN: C_BEN, ARM_ADV: C_ADV}


def style():
    plt.rcParams.update({
        "figure.facecolor": SURF, "axes.facecolor": SURF,
        "savefig.facecolor": SURF, "axes.edgecolor": AXIS,
        "axes.labelcolor": INK2, "text.color": INK,
        "xtick.color": INK2, "ytick.color": INK2,
        "grid.color": GRID, "grid.linewidth": 0.7,
        "axes.grid": True, "axes.axisbelow": True,
        "axes.spines.top": False, "axes.spines.right": False,
        "font.size": 9.5, "axes.titlesize": 9.5,
        "xtick.labelsize": 8.5, "ytick.labelsize": 8.5,
        "legend.frameon": False,
    })


def _densite(ax, d, couleur):
    """32 courbes de tetes + la mediane, par rang."""
    for h in sorted(d.tete.unique()):
        s = d[d.tete == h].sort_values("rang")
        ax.plot(s["lambda"], s.mass_frac, color=couleur, lw=0.45, alpha=0.20)
    med = d.groupby("rang").agg(lam=("lambda", "median"),
                                mf=("mass_frac", "median"))
    ax.plot(med.lam, med.mf, color=couleur, lw=1.6, zorder=5)
    ax.axvline(HIGH_CUT, color=AXIS, lw=0.7, ls=(0, (4, 3)), zorder=1)
    ax.set_yscale("log")
    ax.set_xlim(-0.03, 2.03)
    ax.set_xticks([0.0, 0.5, 1.0, 1.5, 2.0])
    lam, mf = d["lambda"].to_numpy(), d.mass_frac.to_numpy()
    tot = mf.sum()
    return 100 * mf[lam >= HIGH_CUT].sum() / tot if tot > 0 else np.nan


def fig_regions(P, pid, couche, regions, out, tag):
    """2 bras x 6 regions, a la couche declaree. Vue 'propre'."""
    g = P[(P.pair_id == pid) & (P.couche == couche)]
    regs = [r for r in regions if (g.region == r).any()]
    fig, axs = plt.subplots(2, len(regs), figsize=(2.30 * len(regs) + 0.9, 6.2),
                            sharex=True, sharey=True, squeeze=False)
    for i, arm in enumerate((ARM_BEN, ARM_ADV)):
        for j, r in enumerate(regs):
            d = g[(g.arm == arm) & (g.region == r)]
            ax = axs[i][j]
            if d.empty:
                ax.set_visible(False)
                continue
            frac = _densite(ax, d, COUL[arm])
            n = int(d.n_region.iloc[0])
            ax.set_title(f"{r}   n={n}\n$\\lambda \\geq 1$ : {frac:.0f} %",
                         loc="left", fontsize=9.0)
            if j == 0:
                ax.set_ylabel(f"{NOM_BRAS[arm]}\n$m_k / \\sum_j m_j$",
                              fontsize=9.0)
            if i == 1:
                ax.set_xlabel("$\\lambda_k$")
    fig.suptitle(f"Spectre d'energie sur le SOUS-GRAPHE INDUIT de chaque "
                 f"region  —  couche {couche}  —  {tag}  —  paire {pid}",
                 x=0.010, ha="left", fontsize=11.5, color=INK)
    fig.text(0.010, 0.012,
             "Chaque region est projetee sur les vecteurs propres de son "
             "propre sous-graphe. Aucune valeur propre commune imposee ; "
             "echelle [0, 2] partout. Les aretes inter-regions sont exclues "
             "par construction.", fontsize=7.8, color=MUTED, ha="left")
    fig.tight_layout(rect=(0, 0.035, 1, 0.935))
    p = os.path.join(out, f"spectre_regions__{tag}__regions_couche{couche}.png")
    fig.savefig(p, dpi=150)
    plt.close(fig)
    return p


def fig_grille(P, pid, region, arm, out, tag):
    """Une couche par panneau, pour une region. Analogue de propre_grille."""
    g = P[(P.pair_id == pid) & (P.arm == arm) & (P.region == region)]
    layers = sorted(g.couche.unique())
    if not layers:
        return None
    col = COUL[arm]
    nc = 4
    nr = int(np.ceil(len(layers) / nc))
    fig, axs = plt.subplots(nr, nc, figsize=(13.0, 2.75 * nr),
                            sharex=True, sharey=True)
    axs = np.atleast_1d(axs).ravel()
    for i, l in enumerate(layers):
        frac = _densite(axs[i], g[g.couche == l], col)
        axs[i].set_title(f"couche {l}   $\\lambda \\geq 1$ : {frac:.0f} %",
                         loc="left", fontsize=9.5)
    for j in range(len(layers), axs.size):
        axs[j].set_visible(False)
    for j in range(axs.size - nc, axs.size):
        if axs[j].get_visible():
            axs[j].set_xlabel("$\\lambda_k$")
    for j in range(0, axs.size, nc):
        axs[j].set_ylabel("$m_k / \\sum_j m_j$")
    n = int(g.n_region.iloc[0])
    fig.suptitle(f"Densite spectrale couche par couche  —  region "
                 f"'{region}' (n={n}), sous-graphe induit  —  "
                 f"{NOM_BRAS[arm]}  —  {tag}  —  paire {pid}",
                 x=0.012, ha="left", fontsize=12, color=INK)
    fig.tight_layout(rect=(0, 0, 1, 1 - 0.055 * 4 / nr))
    sfx = "benin" if arm == ARM_BEN else "adverse"
    p = os.path.join(out, f"spectre_regions__{tag}__grille_{sfx}.png")
    fig.savefig(p, dpi=150)
    plt.close(fig)
    return p


def _cumulee(d):
    """F(lambda) exacte sur EVAL_GRID. -> (cles, matrice [cellules, grille]).

    Les valeurs propres sont ramenees dans [0, 2] : le mode trivial vaut
    -1e-17 dans un bras et +1e-17 dans l'autre, et une evaluation en
    lambda = 0 compterait ce mode d'un cote seulement -- un artefact du
    signe du zero qui vaut a lui seul la masse du mode trivial.
    """
    cles, out = [], []
    for k, c in d.groupby(["pair_id", "tete"], sort=True):
        c = c.sort_values("lambda")
        lam = np.clip(c["lambda"].to_numpy(np.float64), 0.0, 2.0)
        cum = np.concatenate(([0.0], np.cumsum(c.mass_frac.to_numpy())))
        cles.append(k)
        out.append(cum[np.searchsorted(lam, EVAL_GRID, side="right")])
    return (cles, np.asarray(out) if out
            else np.zeros((0, EVAL_GRID.size)))


def fig_cumulative(P, couche, regions, out, tag, n_paires):
    """F_R(lambda), une region par panneau. Seule comparaison licite."""
    g = P[P.couche == couche]
    regs = [r for r in regions if (g.region == r).any()]
    nc = min(3, len(regs))
    nr = int(np.ceil(len(regs) / nc))
    fig, axs = plt.subplots(nr, nc, figsize=(4.3 * nc, 3.5 * nr),
                            sharex=True, sharey=True, squeeze=False)
    axs = axs.ravel()
    for i, r in enumerate(regs):
        ax = axs[i]
        txt, meds, cur = [], {}, {}
        for arm in (ARM_BEN, ARM_ADV):
            cles, C = _cumulee(g[(g.arm == arm) & (g.region == r)])
            if C.size == 0:
                continue
            cur[arm] = (cles, C)
            med = np.median(C, axis=0)
            meds[arm] = med
            lo, hi = np.percentile(C, [25, 75], axis=0)
            ax.fill_between(EVAL_GRID, lo, hi, color=COUL[arm], alpha=0.16,
                            lw=0)
            # Le bras benin est trace PLUS EPAIS et dessous : la ou les deux
            # courbes coincident exactement (regions anterieures au slot), un
            # lisere bleu reste visible sous l'orange. Sans cela le lecteur
            # croit que le bras benin manque.
            ax.plot(EVAL_GRID, med, color=COUL[arm],
                    lw=3.4 if arm == ARM_BEN else 1.6,
                    zorder=3 if arm == ARM_BEN else 4, label=NOM_BRAS[arm],
                    solid_capstyle="butt")
            j = int(np.searchsorted(EVAL_GRID, HIGH_CUT))
            txt.append(f"{100 * (1 - med[j]):.0f} %")
        ax.axvline(HIGH_CUT, color=AXIS, lw=0.8, ls=(0, (4, 3)), zorder=1)
        ax.set_xlim(0, 2)
        ax.set_ylim(-0.02, 1.02)
        # Distance PAR CELLULE, appariee (paire, tete), puis mediane. Le
        # maximum entre les deux courbes MEDIANES serait trompeur : la ou
        # la cumulee fait une marche, un decalage minime de la position de
        # la marche produit un ecart enorme sans rien signifier.
        # D exacte, sur la reunion des valeurs propres des deux cellules :
        # le meme calcul que analyse_region_spectrum.py, pas une version
        # sur grille. Une grille de pas 0,005 manque les intervalles entre
        # valeurs propres voisines et sous-estime D.
        marque = ""
        gr = g[g.region == r]
        A = _cles(gr[gr.arm == ARM_ADV])
        B = _cles(gr[gr.arm == ARM_BEN])
        Dc = [_kolmogorov(A[k][0], A[k][1], B[k][0], B[k][1])
              for k in (set(A) & set(B)) if A[k][0].size == B[k][0].size]
        if Dc:
            marque = ("   identiques (causalite)" if max(Dc) == 0.0
                      else f"   D median {np.median(Dc):.3f}")
        ax.set_title(f"{r}    $\\lambda \\geq 1$ : " + "  vs  ".join(txt)
                     + marque, loc="left", fontsize=8.6)
        if i % nc == 0:
            ax.set_ylabel("$F(\\lambda)$  energie cumulee")
        if i >= len(regs) - nc:
            ax.set_xlabel("$\\lambda$")
    for j in range(len(regs), axs.size):
        axs[j].set_visible(False)
    axs[0].legend(fontsize=8.6, loc="upper left", labelcolor=INK2)
    fig.suptitle(f"Energie cumulee par region, sous-graphe induit  —  "
                 f"couche {couche}  —  {tag}  —  {n_paires} paires x 32 tetes",
                 x=0.010, ha="left", fontsize=11.5, color=INK)
    fig.text(0.010, 0.012,
             "Fonction en escalier exacte, aucune interpolation. La cumulee "
             "est la seule comparaison licite entre regions : elles n'ont ni "
             "le meme nombre de tokens ni le meme nombre de modes. Bande = "
             "quartiles sur les cellules.", fontsize=7.8, color=MUTED,
             ha="left")
    fig.tight_layout(rect=(0, 0.035, 1, 0.945))
    p = os.path.join(out, f"spectre_regions__{tag}__cumulative.png")
    fig.savefig(p, dpi=150)
    plt.close(fig)
    return p


def fig_partagee(S, pid, couche, regions, out, tag):
    """Graphe ENTIER, signal restreint : memes frequences pour les 6."""
    g = S[(S.pair_id == pid) & (S.couche == couche)]
    regs = [r for r in regions if (g.region == r).any()]
    fig, axs = plt.subplots(2, len(regs), figsize=(2.30 * len(regs) + 0.9, 6.2),
                            sharex=True, sharey=True, squeeze=False)
    for i, arm in enumerate((ARM_BEN, ARM_ADV)):
        for j, r in enumerate(regs):
            d = g[(g.arm == arm) & (g.region == r)]
            ax = axs[i][j]
            if d.empty:
                ax.set_visible(False)
                continue
            frac = _densite(ax, d, COUL[arm])
            n = int(d.n_region.iloc[0])
            ax.set_title(f"{r}   n={n}\n$\\lambda \\geq 1$ : {frac:.0f} %",
                         loc="left", fontsize=9.0)
            if j == 0:
                ax.set_ylabel(f"{NOM_BRAS[arm]}\n$m_k / \\sum_j m_j$",
                              fontsize=9.0)
            if i == 1:
                ax.set_xlabel("$\\lambda_k$")
    fig.suptitle(f"Signal restreint a la region, GRAPHE ENTIER  —  couche "
                 f"{couche}  —  {tag}  —  paire {pid}",
                 x=0.010, ha="left", fontsize=11.5, color=INK)
    fig.text(0.010, 0.012,
             "Vue complementaire : memes vecteurs propres du graphe complet "
             "pour les six regions, seul le signal est restreint. Les aretes "
             "inter-regions sont ici conservees.", fontsize=7.8, color=MUTED,
             ha="left")
    fig.tight_layout(rect=(0, 0.035, 1, 0.935))
    p = os.path.join(out, f"spectre_regions__{tag}__partagee_couche{couche}.png")
    fig.savefig(p, dpi=150)
    plt.close(fig)
    return p


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=os.path.join("results", "opi",
                                                   "region_spectrum"))
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
                                            "propre_long.parquet"))])
    for tag in tags:
        d = os.path.join(args.root, tag)
        prov = json.load(open(os.path.join(d, "provenance.json"),
                              encoding="utf-8"))
        couche = int(prov["couche_declaree"])
        region = prov["region_declaree"]
        regions = list(prov["regions"])
        P = pd.read_parquet(os.path.join(d, "propre_long.parquet"))
        S = pd.read_parquet(os.path.join(d, "partagee_long.parquet"))
        M = pd.read_csv(os.path.join(d, "paires.csv"))
        pid = sorted(P.pair_id.unique())[0]
        print(f"--- {tag} : couche {couche}, region declaree '{region}', "
              f"{len(M)} paires, paire illustree {pid}")
        for p in (fig_regions(P, pid, couche, regions, d, tag),
                  fig_grille(P, pid, region, ARM_BEN, d, tag),
                  fig_grille(P, pid, region, ARM_ADV, d, tag),
                  fig_cumulative(P, couche, regions, d, tag, len(M)),
                  fig_partagee(S, pid, couche, regions, d, tag)):
            if p:
                print("   ecrit", p)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
