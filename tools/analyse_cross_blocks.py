"""Les deux tableaux : decomposition par bloc, et specificite a l'injection.

POINT 1. Pour chaque bloc, les trois termes du croisement -- graphe seul,
signal seul, interaction -- sur le contraste C4. On lit bloc par bloc si
l'injection agit en reorientant l'attention ou en modifiant le contenu.

POINT 2. Le meme tableau sur le contraste C3_wiki, deux bras BENINS. Le
rapport des deux dit si un terme est propre a l'injection ou se produit des
qu'on change le texte du slot.

AGREGATION. Delta CELLULE PAR CELLULE (le script de mesure l'a deja fait),
puis MEDIANE par paire, puis test apparie sur les paires. Jamais l'inverse.

CORRECTION BH. 21 blocs x 3 termes = 63 tests, corriges SEPAREMENT dans
chaque contraste. Famille hybride, jamais melangee aux metriques
attention-only.

RAPPORT DE SPECIFICITE. ``|dz(C4)| / |dz(C3_wiki)|``, declare comme lecture
principale parce qu'il tient compte de la dispersion ; le rapport des
amplitudes medianes est rendu a cote. Un rapport proche de 1 signifie que
le terme bouge autant entre deux textes benins qu'entre benin et injecte :
il ne dit rien de l'injection.

    python tools/analyse_cross_blocks.py --root results/opi/cross_blocks
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

from analyse_region_energy import FDR_Q, bh_fdr, dz, paired_test

TERMES = ("graphe", "signal", "interaction", "total")
#: Les trois termes additifs. ``total`` est leur somme, donc redondant :
#: il est rapporte mais exclu de la correction BH, ou il compterait deux
#: fois la meme information.
TERMES_BH = ("graphe", "signal", "interaction")
CONTRASTE_ADV = "C4"
CONTRASTE_BEN = "C3_wiki"
#: Plancher sur |dz| du contraste benin, pour que le rapport ne soit pas
#: domine par un denominateur quasi nul. Declare.
DZ_PLANCHER = 0.05

C_G, C_S, C_I = "#2a78d6", "#eb6834", "#6b8f3a"   # graphe / signal / inter
INK, INK2, MUTED = "#0b0b0b", "#52514e", "#898781"
GRID, AXIS, SURF = "#e1e0d9", "#c3c2b7", "#fcfcfb"


def style():
    plt.rcParams.update({
        "figure.facecolor": SURF, "axes.facecolor": SURF,
        "savefig.facecolor": SURF, "font.family": "sans-serif",
        "font.sans-serif": ["Segoe UI", "DejaVu Sans", "sans-serif"],
        "axes.edgecolor": AXIS, "axes.linewidth": 0.8,
        "axes.labelcolor": INK2, "axes.titlecolor": INK,
        "axes.titlesize": 10.5, "axes.labelsize": 9,
        "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6,
        "xtick.color": MUTED, "ytick.color": MUTED,
        "xtick.labelsize": 8, "ytick.labelsize": 8,
        "legend.frameon": False, "legend.fontsize": 8.5,
        "axes.spines.top": False, "axes.spines.right": False})


def table(D: pd.DataFrame, contraste: str) -> pd.DataFrame:
    """Un contraste -> 21 blocs x 4 termes, testes et corriges."""
    d = D[D.contraste == contraste]
    per_pair = d.groupby(["bloc", "pair_id"])[list(TERMES)].median()
    rows = []
    for bl, g in per_pair.groupby(level="bloc"):
        A = g.to_numpy()
        p = paired_test(A)
        e = dz(A)
        for k, t in enumerate(TERMES):
            rows.append({"contraste": contraste, "bloc": bl, "terme": t,
                         "delta_median": float(np.nanmedian(A[:, k])),
                         "dz": float(e[k]), "p": float(p[k]),
                         "paires_positives": float(np.mean(A[:, k] > 0)),
                         "n_paires": A.shape[0]})
    R = pd.DataFrame(rows)
    m = R.terme.isin(TERMES_BH)
    R.loc[m, "bh_q"] = bh_fdr(R.loc[m, "p"].to_numpy())
    R["signif"] = R.bh_q < FDR_Q
    return R


def specificite(Ra: pd.DataFrame, Rb: pd.DataFrame) -> pd.DataFrame:
    """Rapport adverse / benin-benin, terme par terme et bloc par bloc."""
    a = Ra.set_index(["bloc", "terme"])
    b = Rb.set_index(["bloc", "terme"])
    S = pd.DataFrame({
        "dz_adv": a.dz, "dz_ben": b.dz,
        "delta_adv": a.delta_median, "delta_ben": b.delta_median,
        "q_adv": a.bh_q, "signif_adv": a.signif,
        "paires_adv": a.paires_positives}).reset_index()
    S["rapport_dz"] = (S.dz_adv.abs()
                       / S.dz_ben.abs().clip(lower=DZ_PLANCHER))
    S["rapport_delta"] = (S.delta_adv.abs()
                          / S.delta_ben.abs().clip(lower=1e-300))
    return S


def fig_spec(S: pd.DataFrame, out: str, tag: str):
    """Deux panneaux : la specificite, puis les trois termes des blocs forts.

    (a) ``dz`` adverse contre ``dz`` benin-benin. La diagonale est le lieu
    des termes qui bougent AUTANT entre deux textes benins qu'entre benin
    et injecte : un point sur la diagonale n'apprend rien sur l'injection.
    Plus un point s'en ecarte verticalement, plus le terme est propre a
    l'injection.
    """
    S = S[S.terme.isin(TERMES_BH)]
    cols = {"graphe": C_G, "signal": C_S, "interaction": C_I}
    fig, ax = plt.subplots(1, 2, figsize=(13.0, 5.0))

    lim = float(np.nanmax(np.abs(np.r_[S.dz_adv, S.dz_ben]))) * 1.15
    ax[0].plot([-lim, lim], [-lim, lim], color=AXIS, lw=1.0, zorder=1)
    ax[0].plot([-lim, lim], [lim, -lim], color=AXIS, lw=1.0, zorder=1)
    ax[0].axhline(0, color=GRID, lw=0.8, zorder=1)
    ax[0].axvline(0, color=GRID, lw=0.8, zorder=1)
    for t, c in cols.items():
        g = S[S.terme == t]
        ax[0].scatter(g.dz_ben, g.dz_adv, s=46, color=c, label=t,
                      edgecolors=SURF, linewidths=1.2, zorder=4)
    top = S.reindex(S.rapport_dz.sort_values(ascending=False).index).head(6)
    for _, r in top.iterrows():
        ax[0].annotate(f"{r.bloc} · {r.terme}", (r.dz_ben, r.dz_adv),
                       xytext=(7, 5), textcoords="offset points",
                       fontsize=7.5, color=INK2)
    # Les deux diagonales sont le lieu |dz_adv| = |dz_ben| : un terme qui
    # s'y trouve bouge autant entre deux textes benins qu'entre benin et
    # injecte, quel que soit le signe. L'annotation va en bas a gauche,
    # ou aucun point n'est etiquete.
    ax[0].annotate("sur les diagonales : $|d_z|$ egal dans les deux\n"
                   "contrastes, donc aucun gain de specificite",
                   (-lim * 0.97, lim * 0.97), xytext=(8, -4),
                   textcoords="offset points", ha="left", va="top",
                   fontsize=7.5, color=MUTED, linespacing=1.4)
    ax[0].set_xlim(-lim, lim)
    ax[0].set_ylim(-lim, lim)
    ax[0].set_xlabel("$d_z$ du contraste BENIN vs BENIN   "
                     "(C3_wiki $-$ C3_opi)")
    ax[0].set_ylabel("$d_z$ du contraste ADVERSE   (C4 $-$ C3_opi)")
    ax[0].set_title("(a) specificite a l'injection\n"
                    "63 points : 21 blocs x 3 termes", loc="left")
    ax[0].legend(loc="lower right", title=None)

    blocs = (S[S.terme == "graphe"]
             .assign(amp=lambda x: x.bloc.map(
                 S.groupby("bloc").dz_adv.apply(lambda s: s.abs().max())))
             .sort_values("amp", ascending=False).bloc.head(8).tolist())
    y = np.arange(len(blocs))
    w = 0.26
    ax[1].axvline(0.0, color=AXIS, lw=1.0, zorder=2)
    for j, (t, c) in enumerate(cols.items()):
        v = [float(S[(S.bloc == b) & (S.terme == t)].dz_adv.iloc[0])
             for b in blocs]
        sg = [bool(S[(S.bloc == b) & (S.terme == t)].signif_adv.iloc[0])
              for b in blocs]
        ax[1].barh(y + (1 - j) * w, v, height=w * 0.86, color=c, zorder=3,
                   label=t)
        for i, (vv, ss) in enumerate(zip(v, sg)):
            if ss:
                ax[1].annotate("*", (vv, y[i] + (1 - j) * w),
                               xytext=(4 if vv >= 0 else -9, -3),
                               textcoords="offset points", fontsize=10,
                               color=INK2)
    ax[1].set_yticks(y)
    ax[1].set_yticklabels(blocs, fontsize=8)
    # Marge d'une demi-epaisseur de groupe : sans elle la barre extreme du
    # dernier groupe sort du cadre et parait appartenir a une ligne de plus.
    ax[1].set_ylim(y[-1] + 2 * w, y[0] - 2 * w)
    ax[1].grid(axis="y", visible=False)
    ax[1].set_xlabel("$d_z$ apparie, contraste C4 $-$ C3_opi")
    ax[1].set_title("(b) les trois termes, 8 blocs de plus forte amplitude\n"
                    "* = significatif apres BH sur les 63 tests", loc="left")
    ax[1].legend(loc="lower right")

    fig.suptitle(f"Decomposition croisee signal / graphe, bloc par bloc  —  "
                 f"{tag}", x=0.012, ha="left", fontsize=12, color=INK)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    p = os.path.join(out, f"cross_blocks__{tag}.png")
    fig.savefig(p, dpi=170)
    plt.close(fig)
    return p


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=os.path.join("results", "opi",
                                                   "cross_blocks"))
    ap.add_argument("--tag", default=None)
    ap.add_argument("--show", type=int, default=8)
    args = ap.parse_args()
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass
    style()

    tags = ([args.tag] if args.tag else
            [t for t in sorted(os.listdir(args.root))
             if os.path.exists(os.path.join(args.root, t,
                                            "cross_blocks.parquet"))])
    for tag in tags:
        d = os.path.join(args.root, tag)
        prov = json.load(open(os.path.join(d, "provenance.json"),
                              encoding="utf-8"))
        D = pd.read_parquet(os.path.join(d, "cross_blocks.parquet"))
        CK = pd.read_csv(os.path.join(d, "controles.csv"))
        print("=" * 78)
        print(f"{tag}  —  {prov['n_pairs_kept']} paires, "
              f"{prov['n_layers']} couches x {prov['n_heads']} tetes")
        print(f"reference {prov['arm_ref']}, termes normalises par "
              f"{prov['normalisation_terme']}, famille {prov['family']}")
        dx = max(CK[[c for c in CK if c.startswith("max_dX")]].max())
        da = max(CK[[c for c in CK if c.startswith("max_dA")]].max())
        print(f"\nControles d'alignement :")
        print(f"  X_task identique entre bras        max {dx:.2e}")
        print(f"  A[task, task] identique entre bras max {da:.2e}")
        print(f"  blocs avant slot, Delta en 'none'  max "
              f"{CK.max_zero_none_avant_slot.max():.2e}   "
              f"(doit etre nul ; en 'sym' il ne l'est pas, le degre couple "
              f"task au slot)")

        Ra = table(D, CONTRASTE_ADV)
        Rb = table(D, CONTRASTE_BEN)
        S = specificite(Ra, Rb)
        pd.concat([Ra, Rb], ignore_index=True).to_csv(
            os.path.join(d, "tests_par_bloc.csv"), index=False)
        S.to_csv(os.path.join(d, "specificite.csv"), index=False)

        # ---------------- POINT 1 ----------------
        print("\n" + "-" * 78)
        print(f"POINT 1 — decomposition par bloc, contraste "
              f"{CONTRASTE_ADV} - {prov['arm_ref']}")
        print("-" * 78)
        amp = (Ra[Ra.terme.isin(TERMES_BH)].groupby("bloc").dz
               .apply(lambda s: s.abs().max()).sort_values(ascending=False))
        print(f"{'bloc':20s}{'graphe':>12s}{'signal':>12s}"
              f"{'interact':>12s}{'total':>12s}   lecture")
        for bl in amp.head(args.show).index:
            g = Ra[Ra.bloc == bl].set_index("terme")
            vs = {t: g.loc[t, "dz"] for t in TERMES}
            sg = {t: g.loc[t, "signif"] for t in TERMES_BH}
            ag, asg = abs(vs["graphe"]), abs(vs["signal"])
            if max(ag, asg) < 0.3:
                lec = "rien"
            elif ag > 2 * asg:
                lec = "surtout l'attention"
            elif asg > 2 * ag:
                lec = "surtout le contenu"
            else:
                lec = "les deux"
            cells = "".join(
                f"{vs[t]:+11.2f}{'*' if sg.get(t, False) else ' '}"
                for t in TERMES)
            print(f"{bl:20s}{cells}   {lec}")
        print(f"  * = significatif apres BH sur les "
              f"{len(TERMES_BH) * D.bloc.nunique()} tests du contraste "
              f"(q < {FDR_Q}) ; 'total' est la somme des trois, exclu de BH")

        # ---------------- POINT 2 ----------------
        print("\n" + "-" * 78)
        print(f"POINT 2 — specificite : le meme terme entre deux bras BENINS")
        print("-" * 78)
        T = S[S.terme.isin(TERMES_BH)].copy()
        T = T.reindex(T.rapport_dz.sort_values(ascending=False).index)
        print(f"{'bloc':20s}{'terme':13s}{'dz adv':>9s}{'dz benin':>10s}"
              f"{'rapport':>9s}{'paires +':>10s}")
        for _, r in T.head(args.show).iterrows():
            mark = "*" if r.signif_adv else " "
            print(f"{r.bloc:20s}{r.terme:13s}{r.dz_adv:+9.2f}"
                  f"{r.dz_ben:+10.2f}{r.rapport_dz:9.1f}"
                  f"{100 * r.paires_adv:9.0f} %{mark}")
        print(f"\n  les termes les MOINS specifiques (rapport proche de 1) :")
        for _, r in T.tail(4).iterrows():
            print(f"{r.bloc:20s}{r.terme:13s}{r.dz_adv:+9.2f}"
                  f"{r.dz_ben:+10.2f}{r.rapport_dz:9.1f}")
        print(f"  plancher declare sur |dz| benin : {DZ_PLANCHER}")

        p = fig_spec(S, d, tag)
        print(f"\nfigure : {p}")
        print(f"tables : {d}/tests_par_bloc.csv, specificite.csv")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
