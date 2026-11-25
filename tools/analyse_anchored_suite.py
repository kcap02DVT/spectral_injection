"""Analyse de la suite ancree : six modeles, quatre bras, deux ancres, deux
ponderations. Aucun classifieur, aucune AUROC, aucun seuil.

CONTRASTES (Delta apparie par paire, cellule = couche x tete) :
  inj      C4       - C3_opi     injection
  ben      C3_wiki  - C3_opi     autre texte benin
  aln      C3_align - C3_opi     forme d'instruction, sans conflit
  conflit  C4       - C3_align   conflit seul, a forme d'instruction egale
Identite exacte : inj = aln + conflit.

PAR (modele, ancre, poids, contraste, quantite) : cellules retenues par
Benjamini-Hochberg (q = 0,05, par famille deployable / oracle), et part des
cellules retenues ou la quantite AUGMENTE.

QUESTION DECISIVE, declaree : sur les cellules retenues du contraste inj,
quelle part du Delta d'injection l'instruction alignee reproduit-elle ?
  mediane de Delta_aln / Delta_inj
  proche de 1  -> l'effet est celui de la FORME d'instruction
  proche de 0  -> l'effet est celui du CONFLIT (le detournement)

ENTRE MODELES : sens majoritaire des cellules retenues, par quantite.

SUCCES : Delta_inj des paires detournees contre non detournees, Mann-Whitney
bilateral par cellule, BH par famille ; labels ASR des campagnes
region_energy. Puis signe de l'ecart dans chaque type d'attaque.

    python tools/analyse_anchored_suite.py
"""

import json
import os
import sys

import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from analyse_region_energy import bh_fdr, dz, paired_test       # noqa: E402
from anchored_readout import FAM, QUANTITES                     # noqa: E402

RACINE = os.path.join("results", "opi", "anchored_suite")
ASR_RACINES = (os.path.join("results", "opi", "region_energy"),
               os.path.join("results", "opi", "dirichlet"))
CONTRASTES = {"inj": ("C4", "C3_opi"), "ben": ("C3_wiki", "C3_opi"),
              "aln": ("C3_align", "C3_opi"), "conflit": ("C4", "C3_align")}
ORDRE_MODELES = ("qwen2.5-0.5b", "llama3.2-1b", "qwen2.5-1.5b", "gemma-2-2b",
                 "llama3.2-3b", "phi-4-mini")
FDR_Q = 0.05
TOL = 1e-12
FAM_DE = {q: f for f, qs in FAM.items() for q in qs}


def deltas(S, a, b):
    k = ["pair_id", "couche", "tete"]
    A = S[S.arm == a].set_index(k)[QUANTITES].sort_index()
    B = S[S.arm == b].set_index(k)[QUANTITES].sort_index()
    j = A.index.intersection(B.index)
    return (A.loc[j] - B.loc[j]).reset_index()


def tests(Dl, paires):
    out = []
    for fam, qs in FAM.items():
        bloc = []
        for q in qs:
            piv = Dl.pivot_table(index="pair_id", columns=["couche", "tete"],
                                 values=q).reindex(paires)
            if piv.shape[1] == 0:
                continue
            V = piv.to_numpy(np.float64)
            finite = np.isfinite(V).sum(axis=0) >= 5
            nul = np.nanmax(np.abs(np.where(np.isfinite(V), V, 0)), axis=0) <= TOL
            m = finite & ~nul
            p = np.full(V.shape[1], np.nan)
            d = np.full(V.shape[1], np.nan)
            if m.any():
                p[m] = paired_test(V[:, m])
                d[m] = dz(V[:, m])
            for j, (c, h) in enumerate(piv.columns):
                bloc.append({"famille": fam, "quantite": q, "couche": int(c),
                             "tete": int(h), "dz": d[j], "p": p[j]})
        if not bloc:
            continue
        b = pd.DataFrame(bloc)
        b["q_bh"] = np.nan
        mm = b.p.notna()
        if mm.any():
            b.loc[mm, "q_bh"] = bh_fdr(b.loc[mm, "p"].to_numpy())
        b["signif"] = b.q_bh < FDR_Q
        out.append(b)
    return pd.concat(out, ignore_index=True) if out else pd.DataFrame()


def labels_asr(tag):
    for r in ASR_RACINES:
        f = os.path.join(r, tag, "pairs.csv")
        if os.path.exists(f):
            P = pd.read_csv(f)
            if "asr_defined_C4" in P:
                P = P[P.asr_defined_C4.fillna(False).astype(bool)]
                return dict(zip(P.pair_id, P.asr_hijacked_C4.astype(bool)))
    return None


def couplage(Dl, hij):
    d = Dl[Dl.pair_id.isin(hij)].copy()
    d["hij"] = d.pair_id.map(hij)
    d["att"] = d.pair_id.str.split("__").str[0]
    out = []
    for fam, qs in FAM.items():
        bloc = []
        for q in qs:
            for (c, h), g in d.groupby(["couche", "tete"]):
                x = g.loc[g.hij, q].dropna().to_numpy()
                y = g.loc[~g.hij, q].dropna().to_numpy()
                if x.size < 3 or y.size < 3 or np.ptp(np.r_[x, y]) == 0:
                    continue
                e = float(np.median(x) - np.median(y))
                acc = tot = 0
                for _, ga in g.groupby("att"):
                    xa = ga.loc[ga.hij, q].dropna()
                    ya = ga.loc[~ga.hij, q].dropna()
                    if len(xa) and len(ya):
                        tot += 1
                        acc += int(np.sign(xa.median() - ya.median())
                                   == np.sign(e))
                bloc.append({"famille": fam, "quantite": q, "couche": c,
                             "tete": h, "ecart": e, "strates_ok": acc,
                             "strates": tot,
                             "p": mannwhitneyu(x, y, alternative="two-sided"
                                               ).pvalue})
        if bloc:
            b = pd.DataFrame(bloc)
            b["q_bh"] = bh_fdr(b.p.to_numpy())
            b["signif"] = b.q_bh < FDR_Q
            out.append(b)
    return pd.concat(out, ignore_index=True) if out else pd.DataFrame()


def analyse_modele(dd, tag):
    S = pd.read_parquet(os.path.join(dd, "scalaires.parquet"))
    prov = json.load(open(os.path.join(dd, "provenance.json")))
    M = pd.read_csv(os.path.join(dd, "paires.csv"))
    paires = sorted(S.pair_id.unique())
    lignes, decomp, succes = [], [], []
    for (ancre, poids), Sx in S.groupby(["ancre", "poids"]):
        D = {c: deltas(Sx, a, b) for c, (a, b) in CONTRASTES.items()}
        T = {c: tests(D[c], paires) for c in CONTRASTES}
        for c, Tc in T.items():
            if Tc.empty:
                continue
            for q, g in Tc.groupby("quantite", sort=False):
                s = g[g.signif]
                lignes.append({"modele": tag, "ancre": ancre, "poids": poids,
                               "contraste": c, "quantite": q,
                               "famille": FAM_DE[q], "n_cellules": len(g),
                               "bh": len(s),
                               "part_bh": len(s) / max(len(g), 1),
                               "part_hausse": (float((s.dz > 0).mean())
                                               if len(s) else np.nan),
                               "absdz_med": float(g.dz.abs().median())})
        # decomposition, sur les cellules retenues du contraste inj
        Ti = T["inj"]
        for q in QUANTITES:
            s = Ti[(Ti.quantite == q) & Ti.signif]
            if s.empty:
                continue
            cel = set(zip(s.couche, s.tete))
            k = ["pair_id", "couche", "tete"]
            di = D["inj"].set_index(k)[q]
            da = D["aln"].set_index(k)[q]
            m = [(c, h) in cel for _, c, h in di.index]
            di, da = di[m], da.reindex(di[m].index)
            ok = di.abs() > 1e-9
            r = (da[ok] / di[ok])
            # par cellule : mediane sur les paires, puis mediane sur cellules
            rc = r.groupby(level=["couche", "tete"]).median()
            Tc = T["conflit"]
            sc = Tc[(Tc.quantite == q) & Tc.signif]
            sens_inj = dict(zip(zip(s.couche, s.tete), np.sign(s.dz)))
            garde = sum(1 for c, h, z in zip(sc.couche, sc.tete, sc.dz)
                        if (c, h) in sens_inj and np.sign(z) == sens_inj[(c, h)])
            decomp.append({"modele": tag, "ancre": ancre, "poids": poids,
                           "quantite": q, "cellules_inj": len(s),
                           "part_forme_med": float(rc.median()),
                           "part_forme_q25": float(rc.quantile(0.25)),
                           "part_forme_q75": float(rc.quantile(0.75)),
                           "conflit_garde_sens": garde / len(s)})
        hij = labels_asr(tag)
        if hij is not None and ancre == "fin":
            C = couplage(D["inj"], hij)
            if not C.empty:
                C.insert(0, "poids", poids)
                C.insert(0, "modele", tag)
                succes.append(C)
    n_hij = (sum(labels_asr(tag).values()), len(labels_asr(tag))) \
        if labels_asr(tag) else (0, 0)
    return (pd.DataFrame(lignes), pd.DataFrame(decomp),
            pd.concat(succes, ignore_index=True) if succes else pd.DataFrame(),
            prov, M, n_hij)


def figure(R, out):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    C_UP, C_DN = "#eb6834", "#2a78d6"
    INK, INK2, MUTED = "#0b0b0b", "#52514e", "#898781"
    GRID, AXIS, SURF = "#e1e0d9", "#c3c2b7", "#fcfcfb"
    modeles = [m for m in ORDRE_MODELES if m in set(R.modele)]
    lignes = [("fin", "attn"), ("fin", "vw"), ("finE", "attn")]
    cols = [("inj", "injection\nC4 - C3_opi"),
            ("aln", "instruction alignee\nC3_align - C3_opi"),
            ("conflit", "conflit seul\nC4 - C3_align")]
    noms_l = {("fin", "attn"): "ancre fin du prompt\nattention brute",
              ("fin", "vw"): "ancre fin du prompt\nponderee par les valeurs",
              ("finE", "attn"): "ancre fin du bloc externe\nattention brute"}
    fig, axs = plt.subplots(len(lignes), len(cols),
                            figsize=(4.3 * len(cols) + 1.6, 3.5 * len(lignes)),
                            squeeze=False)
    qs = QUANTITES
    for i, (an, po) in enumerate(lignes):
        for j, (c, titre) in enumerate(cols):
            ax = axs[i][j]
            ax.set_facecolor(SURF)
            for s in ax.spines.values():
                s.set_visible(False)
            g = R[(R.ancre == an) & (R.poids == po) & (R.contraste == c)]
            if g.empty:
                ax.text(0.5, 0.5, "non mesure", ha="center", va="center",
                        color=MUTED, transform=ax.transAxes, fontsize=9)
                ax.set_xticks([])
                ax.set_yticks([])
            for xi, m in enumerate(modeles):
                for yi, q in enumerate(qs):
                    r = g[(g.modele == m) & (g.quantite == q)]
                    if r.empty:
                        continue
                    r = r.iloc[0]
                    if r.part_bh < 0.05 or not np.isfinite(r.part_hausse):
                        ax.scatter(xi, yi, s=26, facecolor=SURF,
                                   edgecolor=AXIS, linewidth=1.0, zorder=3)
                        continue
                    col = C_UP if r.part_hausse > 0.5 else C_DN
                    ax.scatter(xi, yi, s=30 + 330 * r.part_bh, color=col,
                               edgecolor=SURF, linewidth=1.2, zorder=3,
                               alpha=0.35 + 0.65 * abs(r.part_hausse - 0.5) * 2)
            ax.set_xticks(range(len(modeles)))
            ax.set_xticklabels(modeles, rotation=40, ha="right", fontsize=7.8,
                               color=INK2)
            ax.set_yticks(range(len(qs)))
            ax.set_yticklabels(qs if j == 0 else [""] * len(qs), fontsize=8.2,
                               color=INK)
            ax.set_xlim(-0.6, len(modeles) - 0.4)
            ax.set_ylim(len(qs) - 0.5, -0.5)
            ax.grid(True, color=GRID, lw=0.6, zorder=0)
            if i == 0:
                ax.set_title(titre, fontsize=9.6, color=INK, pad=8)
            if j == 0:
                ax.set_ylabel(noms_l[(an, po)], fontsize=8.8, color=INK2,
                              labelpad=8)
    leg = [Line2D([], [], marker="o", ls="", ms=9, color=C_UP,
                  label="la quantite augmente"),
           Line2D([], [], marker="o", ls="", ms=9, color=C_DN,
                  label="la quantite diminue"),
           Line2D([], [], marker="o", ls="", ms=6, markerfacecolor=SURF,
                  markeredgecolor=AXIS, label="moins de 5 % des cellules")]
    fig.legend(handles=leg, loc="lower center", ncol=3, frameon=False,
               fontsize=8.6, labelcolor=INK2, bbox_to_anchor=(0.5, 0.005))
    fig.suptitle("Lectures ancrees : sens et etendue de l'effet, six modeles",
                 x=0.012, ha="left", fontsize=12, color=INK)
    fig.text(0.012, 0.955,
             "Taille = part des tetes retenues par Benjamini-Hochberg ; "
             "couleur = sens majoritaire ; opacite = coherence du sens.",
             fontsize=8.2, color=MUTED, ha="left")
    fig.subplots_adjust(left=0.13, right=0.99, top=0.89, bottom=0.12,
                        wspace=0.12, hspace=0.55)
    fig.savefig(out, dpi=160, facecolor=SURF)
    plt.close(fig)
    return out


def main() -> int:
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass
    racine = sys.argv[1] if len(sys.argv) > 1 else RACINE
    tags = [t for t in ORDRE_MODELES + tuple(sorted(os.listdir(racine)))
            if os.path.isfile(os.path.join(racine, t, "scalaires.parquet"))]
    tags = list(dict.fromkeys(tags))
    allR, allD, allS = [], [], []
    for tag in tags:
        R, Dc, Sc, prov, M, (nh, nd) = analyse_modele(os.path.join(racine, tag),
                                                       tag)
        allR.append(R), allD.append(Dc)
        if not Sc.empty:
            allS.append(Sc)
        print("=" * 100)
        print(f"{tag} : {prov['n_pairs_kept']} paires ; {prov['n_layers']}x"
              f"{prov['n_heads']} ; ponderations {prov['poids']} ; "
              f"C3_align {M.align_kind.value_counts().to_dict()}, part consigne "
              f"mediane {M.part_consigne.median():.2f} ; ASR {nh}/{nd} detournees")
        for (an, po), g in R.groupby(["ancre", "poids"]):
            print(f"\n  [{an} / {po}]   cellules BH (en %) et sens   "
                  f"(+ hausse / - baisse, part des BH qui montent)")
            piv = g.pivot_table(index="quantite", columns="contraste",
                                values=["part_bh", "part_hausse"])
            print(f"  {'quantite':<11}" + "".join(f"{c:>17}" for c in CONTRASTES))
            for q in QUANTITES:
                if q not in piv.index:
                    continue
                cells = []
                for c in CONTRASTES:
                    try:
                        pb = piv.loc[q, ("part_bh", c)]
                        ph = piv.loc[q, ("part_hausse", c)]
                        cells.append(f"{pb:>8.0%} {('' if not np.isfinite(ph) else ('+' if ph > .5 else '-') + f'{ph:.0%}'):>7}")
                    except KeyError:
                        cells.append(f"{'':>16}")
                print(f"  {q:<11}" + " ".join(cells))
        print(f"\n  decomposition (cellules retenues du contraste inj) : part du "
              f"Delta reproduite par l'instruction alignee")
        if not Dc.empty:
            for (an, po), g in Dc.groupby(["ancre", "poids"]):
                print(f"   [{an}/{po}] " + "  ".join(
                    f"{r.quantite}={r.part_forme_med:+.2f}"
                    for _, r in g.iterrows()))
    R = pd.concat(allR, ignore_index=True)
    Dc = pd.concat(allD, ignore_index=True)
    R.to_csv(os.path.join(racine, "synthese.csv"), index=False)
    Dc.to_csv(os.path.join(racine, "decomposition.csv"), index=False)
    print("=" * 100)
    print("CONCORDANCE ENTRE MODELES, sens majoritaire des cellules retenues "
          "(+ hausse, - baisse, . moins de 5 % de cellules)")
    for (an, po, c), g in R.groupby(["ancre", "poids", "contraste"]):
        mods = [m for m in ORDRE_MODELES if m in set(g.modele)]
        print(f"\n  [{an} / {po} / {c}]   " + " ".join(f"{m[:10]:>11}" for m in mods))
        for q in QUANTITES:
            sym = []
            for m in mods:
                r = g[(g.modele == m) & (g.quantite == q)]
                if r.empty:
                    sym.append(f"{'':>11}")
                    continue
                r = r.iloc[0]
                s = "." if (r.part_bh < .05 or not np.isfinite(r.part_hausse)) \
                    else ("+" if r.part_hausse > .5 else "-")
                sym.append(f"{s + f' {r.part_bh:.0%}':>11}")
            ss = [x.strip()[0] for x in sym if x.strip()]
            act = [x for x in ss if x in "+-"]
            verdict = (f"{max(act.count('+'), act.count('-'))}/{len(act)} meme sens"
                       if act else "")
            print(f"  {q:<11}{' '.join(sym)}   {verdict}")
    if allS:
        S = pd.concat(allS, ignore_index=True)
        S.to_csv(os.path.join(racine, "couplage_succes.csv"), index=False)
        print("=" * 100)
        print("COUPLAGE AU SUCCES (ancre fin) : cellules retenues, et part de "
              "celles dont l'ecart garde son signe dans TOUTES les strates "
              "testables")
        for (m, po), g in S.groupby(["modele", "poids"]):
            sig = g[g.signif]
            tout = sig[sig.strates_ok == sig.strates]
            par = sig.groupby("quantite").size().to_dict()
            sens = sig.groupby("quantite").ecart.apply(
                lambda e: "+" if (e > 0).mean() > .5 else "-").to_dict()
            print(f"  {m:<14}{po:<5} {len(sig):>4} cellules ; stables par strate "
                  f"{len(tout)} ; " + ", ".join(f"{q}{sens[q]}{n}"
                                               for q, n in par.items()))
    print(f"\nfigure : {figure(R, os.path.join(racine, 'suite_ancree.png'))}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
