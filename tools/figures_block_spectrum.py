"""Figures du spectre sur l'union de deux regions.

FIGURE 1, ``bloc_declare`` -- le bloc sur lequel porte la question.
  (a) energie cumulee des TROIS bras : reference, injecte, autre texte
      benin. Le troisieme est la seule facon de lire le deuxieme.
  (b) densite spectrale, 32 tetes, un panneau par bras.

FIGURE 2, ``blocs`` -- les quinze blocs ensemble, pour situer celui-la.
  (a) cellules retenues par Benjamini-Hochberg, injection contre benin.
  (b) poids des aretes croisees dans le sous-graphe, contre le
      deplacement du spectre quand on les annule. Un bloc en bas a gauche
      est un bloc dont le spectre d'union n'apprend rien de plus que les
      deux regions separees.

Palette validee (scripts/validate_palette.js, mode clair) :
bleu #2a78d6 / orange #eb6834 / violet #6b4fbb, pire cas protan 24,7.
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

ARM_REF, ARM_ADV, ARM_BEN = "C3_opi", "C4", "C3_wiki"
HIGH_CUT = 1.0
EVAL_GRID = np.linspace(0.0, 2.0, 801)

C_REF, C_ADV, C_WIKI = "#2a78d6", "#eb6834", "#6b4fbb"
COUL = {ARM_REF: C_REF, ARM_ADV: C_ADV, ARM_BEN: C_WIKI}
NOM = {ARM_REF: "reference  (C3_opi)", ARM_ADV: "injecte  (C4)",
       ARM_BEN: "autre benin  (C3_wiki)"}
INK, INK2, MUTED = "#0b0b0b", "#52514e", "#898781"
GRID, AXIS, SURF = "#e1e0d9", "#c3c2b7", "#fcfcfb"


def style():
    plt.rcParams.update({
        "figure.facecolor": SURF, "axes.facecolor": SURF,
        "savefig.facecolor": SURF, "axes.edgecolor": AXIS,
        "axes.labelcolor": INK2, "text.color": INK,
        "xtick.color": INK2, "ytick.color": INK2,
        "grid.color": GRID, "grid.linewidth": 0.7, "axes.grid": True,
        "axes.axisbelow": True, "axes.spines.top": False,
        "axes.spines.right": False, "font.size": 9.5,
        "axes.titlesize": 9.5, "xtick.labelsize": 8.5,
        "ytick.labelsize": 8.5, "legend.frameon": False})


def _cum(d):
    out = []
    for _, c in d.groupby(["pair_id", "tete"], sort=True):
        c = c.sort_values("lambda")
        lam = np.clip(c["lambda"].to_numpy(np.float64), 0.0, 2.0)
        cum = np.concatenate(([0.0], np.cumsum(c.mass_frac.to_numpy())))
        out.append(cum[np.searchsorted(lam, EVAL_GRID, side="right")])
    return np.asarray(out) if out else np.zeros((0, EVAL_GRID.size))


def fig_bloc(P, bloc, couche, out, tag, n_paires, poids, Dsans):
    g = P[(P.bloc == bloc) & (P.couche == couche)]
    fig = plt.figure(figsize=(13.4, 5.3))
    gs = fig.add_gridspec(1, 4, width_ratios=[1.45, 1, 1, 1], wspace=0.26)
    A = fig.add_subplot(gs[0, 0])
    for arm in (ARM_REF, ARM_ADV, ARM_BEN):
        C = _cum(g[g.arm == arm])
        if C.size == 0:
            continue
        med = np.median(C, axis=0)
        lo, hi = np.percentile(C, [25, 75], axis=0)
        A.fill_between(EVAL_GRID, lo, hi, color=COUL[arm], alpha=0.14, lw=0)
        A.plot(EVAL_GRID, med, color=COUL[arm],
               lw=3.2 if arm == ARM_REF else 1.7,
               zorder=3 if arm == ARM_REF else 4, label=NOM[arm])
    A.axvline(HIGH_CUT, color=AXIS, lw=0.8, ls=(0, (4, 3)), zorder=1)
    A.set_xlim(0, 2)
    A.set_ylim(-0.02, 1.02)
    A.set_xlabel("$\\lambda$")
    A.set_ylabel("$F(\\lambda)$  energie cumulee")
    A.set_title("(a)  energie cumulee, trois bras", loc="left", fontsize=10)
    A.legend(fontsize=8.4, loc="upper left", labelcolor=INK2)

    axs = [fig.add_subplot(gs[0, i]) for i in (1, 2, 3)]
    for ax, arm in zip(axs, (ARM_REF, ARM_ADV, ARM_BEN)):
        d = g[g.arm == arm]
        for h in sorted(d.tete.unique()):
            s = d[d.tete == h].sort_values("rang")
            ax.plot(s["lambda"], s.mass_frac, color=COUL[arm], lw=0.4,
                    alpha=0.16)
        med = d.groupby("rang").agg(lam=("lambda", "median"),
                                    mf=("mass_frac", "median"))
        ax.plot(med.lam, med.mf, color=COUL[arm], lw=1.6, zorder=5)
        ax.axvline(HIGH_CUT, color=AXIS, lw=0.7, ls=(0, (4, 3)), zorder=1)
        ax.set_yscale("log")
        ax.set_xlim(-0.03, 2.03)
        ax.set_xticks([0.0, 0.5, 1.0, 1.5, 2.0])
        ax.set_xlabel("$\\lambda_k$")
        lam, mf = d["lambda"].to_numpy(), d.mass_frac.to_numpy()
        ax.set_title(f"{NOM[arm]}\n$\\lambda \\geq 1$ : "
                     f"{100 * mf[lam >= HIGH_CUT].sum() / mf.sum():.0f} %",
                     loc="left", fontsize=8.8)
    axs[0].set_ylabel("$m_k / \\sum_j m_j$")
    for ax in axs[1:]:
        ax.set_yticklabels([])
    axs[0].text(0.03, 0.03, "(b)  densite, 32 tetes", transform=axs[0].transAxes,
                fontsize=10, color=INK, va="bottom", ha="left")

    n = int(g.n_union.iloc[0]) if len(g) else 0
    fig.suptitle(f"Spectre d'energie sur l'union '{bloc}'  (n={n} tokens)  —  "
                 f"couche {couche}  —  {tag}  —  {n_paires} paires x 32 tetes",
                 x=0.010, ha="left", fontsize=11.5, color=INK)
    fig.text(0.010, 0.016,
             f"Le sous-graphe induit sur l'union CONSERVE les aretes "
             f"croisees entre les deux regions. Elles pesent "
             f"{poids:.1%} de la masse du sous-graphe ; les annuler deplace "
             f"le spectre de D = {Dsans:.3f}.",
             fontsize=7.8, color=MUTED, ha="left")
    fig.subplots_adjust(left=0.055, right=0.99, top=0.845, bottom=0.145)
    p = os.path.join(out, f"spectre_bloc__{tag}__{bloc.replace('+', '_')}.png")
    fig.savefig(p, dpi=170)
    plt.close(fig)
    return p


def fig_blocs(R, Dc, bd, couche, out, tag):
    fig, ax = plt.subplots(1, 2, figsize=(13.2, 5.8),
                           gridspec_kw={"wspace": 0.30,
                                        "width_ratios": [1.1, 1]})
    R = R.sort_values("sig_adv", ascending=True).reset_index(drop=True)
    y = np.arange(len(R))
    w = 0.38
    ax[0].barh(y + w / 2, R.sig_adv, height=w, color=C_ADV,
               edgecolor=SURF, linewidth=1.0, label="contraste injection")
    ax[0].barh(y - w / 2, R.sig_ben, height=w, color=C_WIKI,
               edgecolor=SURF, linewidth=1.0, label="contraste benin-benin")
    ax[0].set_yticks(y)
    ax[0].set_yticklabels([f"{b}" + ("  <<" if b == bd else "")
                           for b in R.bloc], fontsize=8.6)
    for t in ax[0].get_yticklabels():
        if t.get_text().endswith("<<"):
            t.set_color(INK)
            t.set_fontweight("bold")
    ax[0].set_xlabel("cellules (couche x tete) retenues a q < 0,05")
    ax[0].set_title("(a)  combien de cellules bougent", loc="left",
                    fontsize=10)
    ax[0].legend(fontsize=8.6, loc="lower right", labelcolor=INK2)
    ax[0].set_ylim(-0.8, len(R) - 0.2)

    d = Dc[(Dc.arm == ARM_REF) & (Dc.couche == couche)]
    g = d.groupby("bloc").agg(poids=("poids_croise", "median"),
                              D=("D_sans_croise", "median"),
                              dl=("d_lambda_sans_croise", "median"))
    for b, r in g.iterrows():
        c = C_ADV if b == bd else MUTED
        ax[1].scatter([r.poids], [r.dl], s=70 if b == bd else 40,
                      facecolor=c, edgecolor=SURF, linewidth=1.2,
                      zorder=4 if b == bd else 3)
    xmax = float(g.poids.max())
    ax[1].set_xlim(-0.012, xmax * 1.12)
    # Les etiquettes des points proches du bord droit sont posees a GAUCHE
    # du point, sinon elles sortent du cadre et se chevauchent.
    lourds = sorted([b for b in g.index if g.loc[b, "poids"] > 0.15],
                    key=lambda b: g.loc[b, "dl"])
    for rang, b in enumerate(lourds):
        dy = (rang - (len(lourds) - 1) / 2) * 11
        ax[1].annotate(b, (g.loc[b, "poids"], g.loc[b, "dl"]),
                       textcoords="offset points", xytext=(-9, dy),
                       fontsize=7.6, color=MUTED, ha="right", va="center")
    if bd in g.index:
        ax[1].annotate(bd, (g.loc[bd, "poids"], g.loc[bd, "dl"]),
                       textcoords="offset points", xytext=(10, 2),
                       fontsize=9, color=INK, ha="left", va="center")
    ax[1].set_xlabel("poids des aretes croisees dans le sous-graphe")
    ax[1].set_ylabel("deplacement du spectre si on les annule\n"
                     "(max $|\\Delta\\lambda|$, median)")
    ax[1].set_title("(b)  l'union apprend-elle quelque chose ?", loc="left",
                    fontsize=10)
    ax[1].annotate("en bas a gauche : le spectre de l'union\n"
                   "ne dit rien de plus que les deux\nregions prises "
                   "separement",
                   xy=(0.035, 0.955), xycoords="axes fraction", fontsize=8.0,
                   color=MUTED, va="top", ha="left", linespacing=1.45)

    for a in ax:
        a.set_facecolor(SURF)
    fig.suptitle(f"Les quinze unions de deux regions  —  couche {couche}  —  "
                 f"{tag}", x=0.010, ha="left", fontsize=11.5, color=INK)
    fig.text(0.010, 0.016,
             "Les quinze blocs sont mesures ; aucun n'est choisi apres "
             "resultats. Wilcoxon apparie bilateral, Benjamini-Hochberg "
             "q<0,05, une seule famille hybride.",
             fontsize=7.8, color=MUTED, ha="left")
    fig.subplots_adjust(left=0.135, right=0.985, top=0.885, bottom=0.135)
    p = os.path.join(out, f"spectre_bloc__{tag}__quinze_blocs.png")
    fig.savefig(p, dpi=170)
    plt.close(fig)
    return p


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=os.path.join("results", "opi",
                                                   "block_spectrum"))
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
                                            "resume_blocs.csv"))])
    for tag in tags:
        d = os.path.join(args.root, tag)
        prov = json.load(open(os.path.join(d, "provenance.json"),
                              encoding="utf-8"))
        couche, bd = int(prov["couche_declaree"]), prov["bloc_declare"]
        P = pd.read_parquet(os.path.join(d, "blocs_long.parquet"))
        R = pd.read_csv(os.path.join(d, "resume_blocs.csv"))
        Dc = pd.read_csv(os.path.join(d, "controle_croise.csv"))
        n = int(pd.read_csv(os.path.join(d, "paires.csv")).shape[0])
        dr = Dc[(Dc.arm == ARM_REF) & (Dc.bloc == bd)
                & (Dc.couche == couche)]
        print(f"--- {tag} : couche {couche}, bloc declare {bd}, {n} paires")
        print("   ecrit", fig_bloc(P, bd, couche, d, tag, n,
                                   float(dr.poids_croise.median()),
                                   float(dr.D_sans_croise.median())))
        print("   ecrit", fig_blocs(R, Dc, bd, couche, d, tag))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
