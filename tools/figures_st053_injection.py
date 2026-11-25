"""Figure de l'essai 0.5.3 : specificite par tete, et amplitude par quantite.

Deux panneaux.

(a) Un point par (quantite, tete) : dz du contraste injecte en abscisse,
    dz du contraste benin-contre-benin en ordonnee. La diagonale y = x est
    la ligne de NON-specificite : un point dessus bouge autant quand on
    remplace le slot par un autre texte benin que par une injection. Plus
    un point est loin de l'axe vertical et proche de l'axe horizontal,
    plus la quantite est specifique a l'injection.

(b) Par quantite, le |dz| median sur les 32 tetes dans les deux
    contrastes, et le nombre de tetes qui passent Benjamini-Hochberg.

Palette validee (scripts/validate_palette.js, mode clair) : le triplet
bleu / orange / vert de mes figures precedentes ECHOUE la separation
protanope (DeltaE 2,7 entre le vert et l'orange). Le vert est remplace
par un violet, pire cas protan 24,7 et tritan 29,4.
"""

import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt                                 # noqa: E402
from matplotlib.lines import Line2D                             # noqa: E402

from test_st053_injection import FAM, QUANTITES                 # noqa: E402

RACINE = os.path.join("results", "opi", "st053")

C_FAM = {"A_attention": "#2a78d6", "B_hybride": "#eb6834",
         "C_valeurs": "#6b4fbb"}
NOM_FAM = {"A_attention": "A  attention seule",
           "B_hybride": "B  hybride (depend de X)",
           "C_valeurs": "C  ponderee par les valeurs"}
INK, INK2, MUTED = "#0b0b0b", "#52514e", "#898781"
GRID, AXIS, SURF = "#e1e0d9", "#c3c2b7", "#fcfcfb"
FAM_DE = {q: f for f, qs in FAM.items() for q in qs}


def _style(ax):
    ax.set_facecolor(SURF)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(AXIS)
        ax.spines[s].set_linewidth(0.8)
    ax.tick_params(colors=INK2, labelsize=8.5, width=0.8, length=3)
    ax.grid(True, color=GRID, lw=0.7, zorder=0)
    ax.set_axisbelow(True)


def figure(Ta: pd.DataFrame, Tb: pd.DataFrame, R: pd.DataFrame,
           out: str, tag: str, n_paires: int) -> str:
    a = Ta.set_index(["quantite", "couche", "tete"])
    b = Tb.set_index(["quantite", "couche", "tete"])
    inter = a.index.intersection(b.index)
    G = pd.DataFrame({"dz_adv": a.loc[inter].dz,
                      "dz_ben": b.loc[inter].dz}).reset_index()
    G["famille"] = G.quantite.map(FAM_DE)

    fig, ax = plt.subplots(1, 2, figsize=(13.2, 6.0), facecolor=SURF,
                           gridspec_kw={"width_ratios": [1.0, 1.15],
                                        "wspace": 0.30})

    # ---------------- (a) specificite, un point par (quantite, tete)
    A = ax[0]
    _style(A)
    m = float(np.nanmax(np.abs(np.r_[G.dz_adv.to_numpy(),
                                     G.dz_ben.to_numpy()]))) * 1.12
    A.plot([-m, m], [-m, m], color=AXIS, lw=1.0, ls=(0, (5, 4)), zorder=1)
    A.plot([-m, m], [m, -m], color=AXIS, lw=1.0, ls=(0, (5, 4)), zorder=1)
    A.axhline(0, color=MUTED, lw=0.9, zorder=1)
    A.axvline(0, color=MUTED, lw=0.9, zorder=1)
    for fam, g in G.groupby("famille"):
        A.scatter(g.dz_adv, g.dz_ben, s=30, facecolor=C_FAM[fam],
                  edgecolor=SURF, linewidth=1.2, alpha=0.9, zorder=3,
                  label=NOM_FAM[fam])
    A.set_xlim(-m, m)
    A.set_ylim(-m, m)
    A.set_xlabel("dz  injection moins benin  (C4 - C3_opi)", fontsize=9.5,
                 color=INK2)
    A.set_ylabel("dz  benin moins benin  (C3_wiki - C3_opi)", fontsize=9.5,
                 color=INK2)
    A.set_title("(a)  un point par quantite et par tete", fontsize=10.5,
                color=INK, loc="left", pad=9)
    A.annotate("sur les diagonales :\nle changement n'est pas\nspecifique a"
               " l'injection", xy=(-m * 0.95, -m * 0.48), fontsize=8.2,
               color=MUTED, va="top", ha="left", linespacing=1.45)
    A.legend(frameon=False, fontsize=8.6, loc="lower right",
             labelcolor=INK2, handletextpad=0.35, borderpad=0.2)

    # ---------------- (b) amplitude par quantite
    B = ax[1]
    _style(B)
    R = R.copy()
    R["quantite"] = pd.Categorical(R.quantite, QUANTITES, ordered=True)
    R = R.sort_values("quantite", ascending=False).reset_index(drop=True)
    med = (G.assign(aa=G.dz_adv.abs(), bb=G.dz_ben.abs())
           .groupby("quantite")[["aa", "bb"]].median())
    y = np.arange(len(R))
    for i, r in R.iterrows():
        va, vb = float(med.loc[r.quantite, "aa"]), float(med.loc[r.quantite, "bb"])
        c = C_FAM[r.famille]
        B.plot([vb, va], [y[i], y[i]], color=c, lw=1.6, alpha=0.45, zorder=2,
               solid_capstyle="round")
        B.scatter([vb], [y[i]], s=46, facecolor=SURF, edgecolor=c,
                  linewidth=1.8, zorder=3)
        B.scatter([va], [y[i]], s=58, facecolor=c, edgecolor=SURF,
                  linewidth=1.2, zorder=4)
    B.set_yticks(y)
    B.set_yticklabels(R.quantite.astype(str), fontsize=8.6, color=INK)
    B.set_xlabel("|dz| median sur les 32 tetes", fontsize=9.5, color=INK2)
    B.set_title("(b)  amplitude, et tetes retenues par Benjamini-Hochberg",
                fontsize=10.5, color=INK, loc="left", pad=9)
    xmax = float(med.max().max()) * 1.46
    B.set_xlim(0, xmax)
    B.set_ylim(-0.9, len(R) - 0.1)
    for i, r in R.iterrows():
        B.text(xmax * 0.995, y[i],
               f"{int(r.tetes_signif_adv)} / {int(r.tetes_signif_ben)}",
               fontsize=8.2, color=INK2, ha="right", va="center")
    B.text(xmax * 0.995, len(R) - 0.45, "tetes BH  inj / benin", fontsize=8.0,
           color=MUTED, ha="right", va="center", style="italic")
    B.legend(handles=[
        Line2D([], [], marker="o", ls="", markersize=7, markerfacecolor=INK2,
               markeredgecolor=SURF, label="contraste injection"),
        Line2D([], [], marker="o", ls="", markersize=7, markerfacecolor=SURF,
               markeredgecolor=INK2, markeredgewidth=1.8,
               label="contraste benin-benin")],
        frameon=False, fontsize=8.6, loc="lower right", labelcolor=INK2,
        handletextpad=0.35, borderpad=0.2)

    fig.suptitle(f"spectral-trust 0.5.3 sur injections OPI  --  {tag}  --  "
                 f"{n_paires} paires appariees, couche declaree, 32 tetes",
                 fontsize=11.5, color=INK, x=0.012, ha="left", y=0.975)
    fig.text(0.012, 0.018,
             "Wilcoxon apparie bilateral, dz de Cohen, Benjamini-Hochberg "
             "q<0,05 dans chaque famille separement. Aucun entrainement, "
             "aucun seuil ajuste, batterie declaree avant mesure.",
             fontsize=7.8, color=MUTED, ha="left")
    fig.subplots_adjust(left=0.065, right=0.985, top=0.885, bottom=0.105)
    fig.savefig(out, dpi=200, facecolor=SURF)
    plt.close(fig)
    return out


def main() -> int:
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass
    racine = sys.argv[1] if len(sys.argv) > 1 else RACINE
    for tag in sorted(os.listdir(racine)):
        dd = os.path.join(racine, tag)
        if not os.path.isfile(os.path.join(dd, "resume.csv")):
            continue
        Ta = pd.read_csv(os.path.join(dd, "par_tete_adv.csv"))
        Tb = pd.read_csv(os.path.join(dd, "par_tete_benin.csv"))
        R = pd.read_csv(os.path.join(dd, "resume.csv"))
        n = int(pd.read_csv(os.path.join(dd, "paires.csv")).shape[0])
        out = os.path.join(dd, f"st053_injection__{tag}.png")
        print("ecrit", figure(Ta, Tb, R, out, tag, n))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
