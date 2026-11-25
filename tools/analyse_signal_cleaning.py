"""Lecture des etapes 1, 3, 4 et 5 du plan.

FAMILLE HYBRIDE, corrigee par BH SEPAREMENT de la famille attention-only.

TROIS CORRECTIONS BH DISTINCTES, declarees, jamais melangees :
  etape 3a  les 6 regions, sur la quantite principale hfshare
  etape 3b  les 21 blocs de Dirichlet, mesure share
  etape 5   les L x H cellules de la carte

MOITIES. Les paires sont triees par identifiant ; les rangs PAIRS
selectionnent, les rangs IMPAIRS confirment. Le partage est deterministe et
ne depend d'aucun resultat. ``K_SELECTION`` cellules sont retenues sur la
moitie de selection, puis testees sur l'autre moitie, BH sur ces K seules.

    python tools/analyse_signal_cleaning.py --root results/opi/signal_cleaning
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

ARM_BEN, ARM_ADV = "C3_opi", "C4"
REGIONS = ("task", "normal", "preamble", "slot", "tail", "template")
K_SELECTION = 10

C_BEN, C_ADV = "#2a78d6", "#eb6834"
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


def paired(df: pd.DataFrame, quantities, index=("pair_id", "couche", "tete")):
    """Delta = C4 - C3_opi, CELLULE PAR CELLULE. Jamais l'inverse."""
    piv = df.pivot_table(index=list(index), columns="arm",
                         values=list(quantities))
    out = {}
    for q in quantities:
        c = piv[(q, ARM_ADV)] - piv[(q, ARM_BEN)]
        out[q] = c
    D = pd.DataFrame(out)
    D.index.names = list(index)
    return D.reset_index()


def test_block(D: pd.DataFrame, quantities, label: str):
    """Wilcoxon apparie + dz sur les paires, BH sur le lot donne."""
    per_pair = D.groupby("pair_id")[list(quantities)].median()
    A = per_pair.to_numpy()
    p = paired_test(A)
    q = bh_fdr(p)
    e = dz(A)
    return pd.DataFrame({"lot": label, "quantite": list(quantities),
                         "delta_median": per_pair.median().to_numpy(),
                         "dz": e, "p": p, "bh_q": q, "signif": q < FDR_Q,
                         "paires_positives": [(per_pair[c] > 0).mean()
                                              for c in quantities],
                         "n_paires": len(per_pair)})


def carte(D: pd.DataFrame, q: str, n_layers: int, n_heads: int):
    """Carte couches x tetes : dz, p apparie sur les paires, BH sur L x H."""
    rows = []
    for (l, h), g in D.groupby(["couche", "tete"]):
        x = g.groupby("pair_id")[q].median().to_numpy()[:, None]
        rows.append({"couche": l, "tete": h, "delta": float(np.median(x)),
                     "dz": float(dz(x)[0]), "p": float(paired_test(x)[0]),
                     "n": x.size})
    C = pd.DataFrame(rows)
    C["bh_q"] = bh_fdr(C.p.to_numpy())
    C["signif"] = C.bh_q < FDR_Q
    return C


def split_half(D: pd.DataFrame, q: str):
    """Selection sur les paires de rang pair, confirmation sur les impaires."""
    pids = sorted(D.pair_id.unique())
    sel, conf = pids[0::2], pids[1::2]
    A = carte(D[D.pair_id.isin(sel)], q, 0, 0)
    top = A.reindex(A.dz.abs().sort_values(ascending=False).index).head(
        K_SELECTION)[["couche", "tete", "dz"]].rename(columns={"dz": "dz_sel"})
    B = D.merge(top[["couche", "tete"]], on=["couche", "tete"])
    rows = []
    for (l, h), g in B[B.pair_id.isin(conf)].groupby(["couche", "tete"]):
        x = g.groupby("pair_id")[q].median().to_numpy()[:, None]
        rows.append({"couche": l, "tete": h, "dz_conf": float(dz(x)[0]),
                     "p_conf": float(paired_test(x)[0])})
    R = top.merge(pd.DataFrame(rows), on=["couche", "tete"])
    R["bh_q_conf"] = bh_fdr(R.p_conf.to_numpy())
    R["signif_conf"] = R.bh_q_conf < FDR_Q
    R["meme_signe"] = np.sign(R.dz_sel) == np.sign(R.dz_conf)
    return R, len(sel), len(conf)


def fig_carte(C: pd.DataFrame, q: str, out: str, tag: str, layer: int):
    """Carte divergente des dz. Deux teintes + gris neutre au milieu."""
    from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm
    M = C.pivot(index="couche", columns="tete", values="dz")
    v = float(np.nanpercentile(np.abs(M.to_numpy()), 98)) or 1.0
    cm = LinearSegmentedColormap.from_list(
        "d", [C_BEN, "#f2f1ec", C_ADV])
    fig, ax = plt.subplots(figsize=(12.0, 4.6))
    im = ax.imshow(M.to_numpy(), cmap=cm, aspect="auto",
                   norm=TwoSlopeNorm(vcenter=0.0, vmin=-v, vmax=v),
                   interpolation="nearest")
    sg = C[C.signif]
    if len(sg):
        ax.scatter(sg.tete, sg.couche, s=14, facecolors="none",
                   edgecolors=INK, linewidths=0.9, zorder=5)
    # Reperee par une fleche dans la marge, pas par un trait en travers :
    # un trait sur la ligne masquerait les cellules qu'il designe.
    ax.annotate(f"couche declaree {layer}  ", (0, layer), xytext=(-46, 0),
                textcoords="offset points", ha="right", va="center",
                color=INK2, fontsize=8, annotation_clip=False,
                arrowprops=dict(arrowstyle="-|>", color=INK2, lw=0.9,
                                shrinkA=2, shrinkB=4))
    ax.set_xlabel("tete")
    ax.set_ylabel("couche")
    ax.grid(False)
    ax.set_title(f"$\\Delta$ apparie de {q}, C4 $-$ C3_opi, par cellule\n"
                 f"cercle = significatif apres BH sur les "
                 f"{len(C)} cellules (q < {FDR_Q})", loc="left")
    cb = fig.colorbar(im, ax=ax, pad=0.012, fraction=0.03)
    cb.set_label("taille d'effet appariee  $d_z$", fontsize=8, color=INK2)
    cb.outline.set_visible(False)
    fig.suptitle(f"Carte couches x tetes apres nettoyage du signal  —  {tag}",
                 x=0.012, ha="left", fontsize=12, color=INK)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    p = os.path.join(out, f"carte_{q}.png")
    fig.savefig(p, dpi=170)
    plt.close(fig)
    return p


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=os.path.join("results", "opi",
                                                   "signal_cleaning"))
    ap.add_argument("--tag", default=None)
    args = ap.parse_args()
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass
    style()

    tags = ([args.tag] if args.tag else
            [t for t in sorted(os.listdir(args.root))
             if os.path.exists(os.path.join(args.root, t, "cells.parquet"))])
    for tag in tags:
        d = os.path.join(args.root, tag)
        prov = json.load(open(os.path.join(d, "provenance.json"),
                              encoding="utf-8"))
        layer = int(prov["primary_layer"])
        VP = prov["primary_variant"]
        RP = prov["primary_region"]
        C = pd.read_parquet(os.path.join(d, "cells.parquet"))
        allrows = []

        print("=" * 78)
        print(f"{tag}  —  {prov['n_pairs_kept']} paires, "
              f"{prov['n_layers']} couches x {prov['n_heads']} tetes")
        print(f"famille {prov['family']}, variante principale {VP}, "
              f"region principale {RP}, couche declaree {layer}")

        # ================= ETAPE 1 =================
        print("\n" + "-" * 78)
        print("ETAPE 1 — nettoyage du signal")
        print("-" * 78)
        g = (C[C.couche == layer].groupby("variante")
             .agg(token_max=("token_share_max", "median"),
                  frac_high=("frac_high", "median"),
                  lissage=("lissage", "median"),
                  hf_slot=("hf_slot", "median"),
                  hfshare_slot=("hfshare_slot", "median"),
                  part_slot=("part_slot", "median")))
        print(f"Couche {layer}, mediane des cellules des deux bras.")
        print(f"{'variante':26s}{'token max':>11s}{'frac_high':>11s}"
              f"{'lissage':>10s}{'hf_slot':>10s}{'hfshare':>10s}"
              f"{'part_slot':>11s}")
        for v, r in g.iterrows():
            print(f"{v:26s}{r.token_max:11.4f}{r.frac_high:11.4f}"
                  f"{r.lissage:10.4f}{r.hf_slot:10.4f}"
                  f"{r.hfshare_slot:10.4f}{r.part_slot:11.4f}")
        g.to_csv(os.path.join(d, "etape1_variantes.csv"))

        tm = float(g.loc[VP, "token_max"])
        seuil = float(prov["seuil_token"])
        print(f"\nRegle conditionnelle declaree : si sous {VP} un seul token "
              f"porte plus de {100 * seuil:.0f} % de ||X||_F^2, la variante "
              f"cosinus devient principale.")
        print(f"  token le plus lourd sous {VP} : {100 * tm:.2f} %  -> "
              f"{'BASCULE vers V4_cosinus' if tm > seuil else 'la regle ne se declenche pas, ' + VP + ' reste principale'}")
        if tm > seuil:
            VP = "V4_cosinus"

        S = pd.read_csv(os.path.join(d, "sinks.csv"))
        print(f"\nPuits d'attention a la couche {layer} : flux entrant moyen "
              f"au-dela de {prov['seuil_puits']:.0f} x l'uniforme.")
        print(f"  nombre de puits par cellule : "
              f"median {S.n_puits_couche.median():.0f}, "
              f"max {S.n_puits_couche.max():.0f}")
        tops = (S.groupby("rang")
                .agg(x_uniforme=("x_uniforme", "median"),
                     est_puits=("est_puits", "mean")).reset_index())
        ex = (S[S.rang <= 3].groupby(["rang", "token"]).size()
              .reset_index(name="n").sort_values(["rang", "n"],
                                                 ascending=[True, False]))
        print(f"  {'rang':>5s}{'x uniforme':>12s}{'% puits':>10s}   "
              f"token le plus frequent a ce rang")
        for _, r in tops.iterrows():
            t = ex[ex.rang == r["rang"]]
            nm = t.token.iloc[0] if len(t) else ""
            print(f"  {int(r['rang']):5d}{r.x_uniforme:12.2f}"
                  f"{100 * r.est_puits:9.0f} %   {nm!r}")

        # ================= ETAPE 3 =================
        print("\n" + "-" * 78)
        print(f"ETAPE 3 — mesure par region, variante {VP}, couche {layer}")
        print("-" * 78)
        CP = C[(C.variante == VP) & (C.couche == layer)]
        for fam, pref in (("hfshare (fraction, somme 1)", "hfshare_"),
                          ("hf (amplification, non bornee)", "hf_")):
            qs = [f"{pref}{r}" for r in REGIONS]
            D = paired(CP, qs)
            R = test_block(D, qs, pref.rstrip("_"))
            allrows.append(R)
            print(f"\n  {fam} — BH sur les {len(REGIONS)} regions")
            print(f"  {'region':12s}{'Delta median':>15s}{'dz':>8s}"
                  f"{'p':>10s}{'q_BH':>9s}{'paires +':>10s}")
            for _, r in R.iterrows():
                mark = "*" if r.signif else " "
                flag = "  <== PRINCIPALE" if r.quantite.endswith(RP) else ""
                print(f" {mark}{r.quantite.replace(pref, ''):12s}"
                      f"{r.delta_median:+15.4e}{r.dz:+8.2f}{r.p:10.3e}"
                      f"{r.bh_q:9.3e}{100 * r.paires_positives:9.0f} %{flag}")

        B = pd.read_parquet(os.path.join(d, "blocks.parquet"))
        Db = (B.pivot_table(index=["pair_id", "tete", "bloc"], columns="arm",
                            values="share"))
        Db = (Db[ARM_ADV] - Db[ARM_BEN]).rename("d").reset_index()
        piv = Db.pivot_table(index="pair_id", columns="bloc", values="d",
                             aggfunc="median")
        A = piv.to_numpy()
        p = paired_test(A)
        RB = pd.DataFrame({"lot": "blocs_share", "quantite": list(piv.columns),
                           "delta_median": np.nanmedian(A, axis=0),
                           "dz": dz(A), "p": p, "bh_q": bh_fdr(p),
                           "signif": bh_fdr(p) < FDR_Q,
                           "paires_positives": (A > 0).mean(axis=0),
                           "n_paires": A.shape[0]})
        allrows.append(RB)
        pb = prov["primary_block"]
        print(f"\n  blocs de Dirichlet sur le signal nettoye, mesure share — "
              f"BH sur les {len(piv.columns)} blocs")
        o = RB.reindex(RB.dz.abs().sort_values(ascending=False).index)
        print(f"  {'bloc':20s}{'Delta median':>15s}{'dz':>8s}{'p':>10s}"
              f"{'q_BH':>9s}{'paires +':>10s}")
        for _, r in o.head(6).iterrows():
            mark = "*" if r.signif else " "
            flag = "  <== PRINCIPAL" if r.quantite == pb else ""
            print(f" {mark}{r.quantite:20s}{r.delta_median:+15.4e}"
                  f"{r.dz:+8.2f}{r.p:10.3e}{r.bh_q:9.3e}"
                  f"{100 * r.paires_positives:9.0f} %{flag}")
        if pb not in o.head(6).quantite.tolist():
            r = RB[RB.quantite == pb].iloc[0]
            mark = "*" if r.signif else " "
            print(f" {mark}{r.quantite:20s}{r.delta_median:+15.4e}"
                  f"{r.dz:+8.2f}{r.p:10.3e}{r.bh_q:9.3e}"
                  f"{100 * r.paires_positives:9.0f} %  <== PRINCIPAL, "
                  f"hors des 6 plus fortes amplitudes")

        # ================= ETAPE 4 =================
        print("\n" + "-" * 78)
        print("ETAPE 4 — d'ou vient l'effet")
        print("-" * 78)
        P = pd.read_parquet(os.path.join(d, "permutation.parquet"))
        print(f"\n  (a) nul par permutation des tokens, graphe FIXE, "
              f"{prov['n_perm']} tirages, couche {layer}")
        print(f"  une mesure insensible a la permutation decrirait le graphe "
              f"et non le contenu")
        print(f"  {'quantite':12s}{'observe':>12s}{'nul moyen':>12s}"
              f"{'z median':>11s}{'|z| > 2':>10s}")
        for nm in ("hf", "hfshare", "frac_high", "lissage"):
            z = P[f"{nm}_z"]
            print(f"  {nm:12s}{P[nm + '_obs'].median():12.4f}"
                  f"{P[nm + '_nul_moy'].median():12.4f}"
                  f"{z.median():+11.2f}{100 * (z.abs() > 2).mean():9.0f} %")

        X = pd.read_parquet(os.path.join(d, "cross.parquet"))
        gr = X.S_ben_sur_graphe_adv - X.S_ben_ben       # graphe seul
        si = X.S_adv_sur_graphe_ben - X.S_ben_ben       # signal seul
        to = X.S_adv_adv - X.S_ben_ben                  # total
        inter = to - gr - si
        print(f"\n  (b) decomposition croisee du lissage, couche {layer}, "
              f"{len(X)} cellules")
        print(f"  Delta total        {to.median():+.4e}   "
              f"({100 * (to > 0).mean():.0f} % de cellules positives)")
        print(f"    graphe seul      {gr.median():+.4e}   "
              f"({100 * (gr > 0).mean():.0f} %)   "
              f"E(X_ben, L_adv) - E(X_ben, L_ben)")
        print(f"    signal seul      {si.median():+.4e}   "
              f"({100 * (si > 0).mean():.0f} %)   "
              f"E(X_adv, L_ben) - E(X_ben, L_ben)")
        print(f"    interaction      {inter.median():+.4e}   "
              f"({100 * (inter > 0).mean():.0f} %)")
        rat = float(np.median(np.abs(si)) /
                    max(np.median(np.abs(gr)), 1e-300))
        print(f"  rapport |signal| / |graphe| (medianes) : {rat:.2f}")
        pd.DataFrame({"graphe_seul": gr, "signal_seul": si,
                      "interaction": inter, "total": to}).to_csv(
            os.path.join(d, "etape4_croisee.csv"), index=False)

        # ================= ETAPE 5 =================
        print("\n" + "-" * 78)
        print("ETAPE 5 — balayage couches x tetes")
        print("-" * 78)
        figs = []
        for q in (f"hfshare_{RP}", "frac_high"):
            Dall = paired(C[C.variante == VP], [q])
            M = carte(Dall, q, prov["n_layers"], prov["n_heads"])
            M.to_csv(os.path.join(d, f"carte_{q}.csv"), index=False)
            ns = int(M.signif.sum())
            print(f"\n  {q} : {ns} cellules sur {len(M)} significatives "
                  f"apres BH (q < {FDR_Q})")
            if ns:
                t = M[M.signif].reindex(
                    M[M.signif].dz.abs().sort_values(ascending=False).index)
                print(f"    {'couche':>7s}{'tete':>6s}{'dz':>8s}"
                      f"{'q_BH':>11s}")
                for _, r in t.head(8).iterrows():
                    print(f"    {int(r.couche):7d}{int(r.tete):6d}"
                          f"{r.dz:+8.2f}{r.bh_q:11.3e}")
            R, nsel, ncon = split_half(Dall, q)
            R.to_csv(os.path.join(d, f"moities_{q}.csv"), index=False)
            print(f"    moities : {nsel} paires selectionnent, {ncon} "
                  f"confirment, {K_SELECTION} cellules retenues")
            print(f"    meme signe en confirmation : "
                  f"{int(R.meme_signe.sum())}/{len(R)}   "
                  f"significatives apres BH : {int(R.signif_conf.sum())}/"
                  f"{len(R)}")
            figs.append(fig_carte(M, q, d, tag, layer))

        pd.concat(allrows, ignore_index=True).to_csv(
            os.path.join(d, "tests.csv"), index=False)
        print(f"\nfigures :")
        for f in figs:
            print(f"  {f}")
        print(f"tables  : {d}/tests.csv, etape1_variantes.csv, "
              f"etape4_croisee.csv, carte_*.csv, moities_*.csv")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
