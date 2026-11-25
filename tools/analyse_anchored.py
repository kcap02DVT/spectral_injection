"""Analyse du pilote des lectures ancrees. Aucun classifieur, aucune AUROC.

PAR CELLULE (couche, tete), pour chaque quantite :
  Delta_inj = C4 - C3_opi,  Delta_ben = C3_wiki - C3_opi, apparies par paire.
  Wilcoxon apparie bilateral, dz de Cohen, Benjamini-Hochberg q = 0,05 DANS
  CHAQUE FAMILLE (deployable, oracle) et dans chaque contraste. Les cellules
  a ecart identiquement nul sont exclues du BH.

PAR QUANTITE, resume declare :
  cellules BH injection / benin, |dz| median, rapport de specificite median
  |dz_inj| / max(|dz_ben|, 0,05), part des cellules BH de dz > 0 (coherence
  de direction), |Delta| median contre Delta median signe (annulation).

ENTRE MODELES : les cellules ne se correspondent pas d'un modele a l'autre.
La concordance porte donc sur le SENS MAJORITAIRE des cellules BH de chaque
quantite. Declare avant de voir les resultats.

COUPLAGE AU SUCCES : a l'interieur du bras C4, paires dont l'ASR est defini,
Delta_inj des paires detournees contre non detournees, Mann-Whitney
bilateral par cellule, BH par famille. Le succes n'est pas assigne au
hasard (il depend du type d'attaque) : la composition des deux groupes est
rendue, et un ecart peut etre un effet de composition.

    python tools/analyse_anchored.py
"""

import json
import os
import sys

import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from analyse_region_energy import bh_fdr, dz, paired_test       # noqa: E402
from anchored_readout import ARM_ADV, ARM_BEN, ARM_REF, FAM     # noqa: E402

RACINE = os.path.join("results", "opi", "anchored")
ASR_RACINE = os.path.join("results", "opi", "dirichlet")
FDR_Q = 0.05
DZ_PLANCHER = 0.05
TOL = 1e-12
FAM_DE = {q: f for f, qs in FAM.items() for q in qs}
QUANTITES = [q for v in FAM.values() for q in v]


def delta(D, alt):
    k = ["pair_id", "couche", "tete"]
    a = D[D.arm == alt].set_index(k)[QUANTITES].sort_index()
    b = D[D.arm == ARM_REF].set_index(k)[QUANTITES].sort_index()
    j = a.index.intersection(b.index)
    return (a.loc[j] - b.loc[j]).reset_index()


def tests(Dl, paires):
    out = []
    for fam, qs in FAM.items():
        bloc = []
        for q in qs:
            piv = Dl.pivot_table(index="pair_id", columns=["couche", "tete"],
                                 values=q).reindex(paires)
            V = piv.to_numpy(np.float64)
            nul = np.nanmax(np.abs(V), axis=0) <= TOL
            p = np.full(V.shape[1], np.nan)
            d = np.full(V.shape[1], np.nan)
            if (~nul).any():
                p[~nul] = paired_test(V[:, ~nul])
                d[~nul] = dz(V[:, ~nul])
            for j, (c, h) in enumerate(piv.columns):
                bloc.append({"famille": fam, "quantite": q, "couche": int(c),
                             "tete": int(h), "dz": d[j], "p": p[j],
                             "abs_delta_med": float(np.nanmedian(np.abs(V[:, j]))),
                             "delta_med": float(np.nanmedian(V[:, j]))})
        b = pd.DataFrame(bloc)
        b["q_bh"] = np.nan
        m = b.p.notna()
        b.loc[m, "q_bh"] = bh_fdr(b.loc[m, "p"].to_numpy())
        b["signif"] = b.q_bh < FDR_Q
        out.append(b)
    return pd.concat(out, ignore_index=True)


def resume(Ta, Tb):
    k = ["quantite", "couche", "tete"]
    a, b = Ta.set_index(k), Tb.set_index(k)
    j = a.index.intersection(b.index)
    G = pd.DataFrame({"famille": a.loc[j].famille, "dz_a": a.loc[j].dz,
                      "dz_b": b.loc[j].dz, "sa": a.loc[j].signif,
                      "sb": b.loc[j].signif,
                      "absd": a.loc[j].abs_delta_med,
                      "d": a.loc[j].delta_med}).reset_index()
    rows = []
    for q in QUANTITES:
        g = G[G.quantite == q]
        sig = g[g.sa]
        rap = g.dz_a.abs() / np.clip(g.dz_b.abs(), DZ_PLANCHER, None)
        rows.append({
            "famille": FAM_DE[q], "quantite": q, "n_cellules": len(g),
            "bh_inj": int(g.sa.sum()), "bh_ben": int(g.sb.sum()),
            "absdz_inj": float(g.dz_a.abs().median()),
            "absdz_ben": float(g.dz_b.abs().median()),
            "rapport_med": float(rap.median()),
            "part_pos_bh": (float((sig.dz_a > 0).mean()) if len(sig)
                            else np.nan),
            "absdz_max": float(g.dz_a.abs().max()),
            "cell_max": (f"L{int(g.loc[g.dz_a.abs().idxmax(), 'couche'])}"
                         f"H{int(g.loc[g.dz_a.abs().idxmax(), 'tete'])}"),
        })
    return pd.DataFrame(rows)


def couplage_asr(Dl, tag):
    f = os.path.join(ASR_RACINE, tag, "pairs.csv")
    if not os.path.exists(f):
        return None, None
    P = pd.read_csv(f)
    P = P[P.asr_defined_C4.fillna(False).astype(bool)]
    hij = dict(zip(P.pair_id, P.asr_hijacked_C4.astype(bool)))
    d = Dl[Dl.pair_id.isin(hij)].copy()
    d["hij"] = d.pair_id.map(hij)
    out = []
    for fam, qs in FAM.items():
        bloc = []
        for q in qs:
            for (c, h), g in d.groupby(["couche", "tete"]):
                x = g.loc[g.hij, q].dropna().to_numpy()
                y = g.loc[~g.hij, q].dropna().to_numpy()
                if x.size < 2 or y.size < 2 or np.ptp(np.r_[x, y]) == 0:
                    continue
                p = mannwhitneyu(x, y, alternative="two-sided").pvalue
                bloc.append({"famille": fam, "quantite": q, "couche": c,
                             "tete": h, "p": p,
                             "ecart_med": float(np.median(x) - np.median(y))})
        b = pd.DataFrame(bloc)
        b["q_bh"] = bh_fdr(b.p.to_numpy())
        b["signif"] = b.q_bh < FDR_Q
        out.append(b)
    comp = (pd.DataFrame({"pair_id": list(hij), "hij": list(hij.values())})
            .assign(attack=lambda x: x.pair_id.str.split("__").str[0]))
    comp = pd.crosstab(comp.attack, comp.hij.map({True: "detournee",
                                                  False: "non"}))
    return pd.concat(out, ignore_index=True), comp


def figure(res, out_png):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    C_INJ, C_BEN, C_ASR = "#eb6834", "#6b4fbb", "#2a78d6"
    INK, INK2, MUTED = "#0b0b0b", "#52514e", "#898781"
    GRID, AXIS, SURF = "#e1e0d9", "#c3c2b7", "#fcfcfb"
    tags = list(res)
    fig, axs = plt.subplots(len(tags), 3, figsize=(15.2, 4.4 * len(tags)),
                            squeeze=False,
                            gridspec_kw={"width_ratios": [1.15, 1.25, 1.0],
                                         "wspace": 0.42, "hspace": 0.70})
    for i, tag in enumerate(tags):
        R, Ta, Tb, Tasr, L = res[tag]
        for a in axs[i]:
            a.set_facecolor(SURF)
            for s in ("top", "right"):
                a.spines[s].set_visible(False)
            for s in ("left", "bottom"):
                a.spines[s].set_color(AXIS)
            a.grid(True, color=GRID, lw=0.7)
            a.set_axisbelow(True)
            a.tick_params(colors=INK2, labelsize=8.4)
        # (a) cellules BH par quantite
        y = np.arange(len(R))[::-1]
        w = 0.38
        A = axs[i][0]
        A.barh(y + w / 2, R.bh_inj, height=w, color=C_INJ, edgecolor=SURF,
               label="injection  (C4 - C3_opi)")
        A.barh(y - w / 2, R.bh_ben, height=w, color=C_BEN, edgecolor=SURF,
               label="benin-benin  (C3_wiki - C3_opi)")
        A.set_yticks(y)
        A.set_yticklabels([f"{q}" + ("   (oracle)" if FAM_DE[q] == "oracle"
                                     else "") for q in R.quantite],
                          fontsize=8.6, color=INK)
        A.set_xlabel(f"cellules retenues a q < 0,05  (sur {int(R.n_cellules.iloc[0])})",
                     color=INK2, fontsize=9)
        A.set_title(f"(a)  {tag} : combien de cellules bougent", loc="left",
                    fontsize=10, color=INK)
        A.legend(fontsize=8.0, loc="upper center", frameon=False,
                 labelcolor=INK2, ncol=2, bbox_to_anchor=(0.45, -0.17))
        # (b) profil en profondeur, famille deployable
        B = axs[i][1]
        dep = [q for q in FAM["deployable"]]
        for T_, c, lab in ((Ta, C_INJ, "injection"), (Tb, C_BEN, "benin-benin")):
            t = T_[T_.quantite.isin(dep)]
            prof = t.groupby("couche").signif.mean().reindex(range(L))
            B.plot(prof.index / max(L - 1, 1), prof.values, color=c, lw=2.0,
                   marker="o", ms=4.5, mec=SURF, mew=1.0, label=lab)
        B.set_xlabel("profondeur relative de la couche", color=INK2, fontsize=9)
        B.set_ylabel("part des cellules retenues", color=INK2, fontsize=9)
        B.set_xlim(-0.02, 1.02)
        B.set_ylim(bottom=0)
        B.set_title("(b)  ou, en profondeur  (famille deployable)",
                    loc="left", fontsize=10, color=INK)
        B.legend(fontsize=8.0, loc="upper right", frameon=False,
                 labelcolor=INK2)
        # (c) couplage au succes
        Cx = axs[i][2]
        if Tasr is not None and len(Tasr):
            cnt = Tasr.groupby("quantite").signif.sum().reindex(R.quantite)
            Cx.barh(y, cnt.values, height=0.6, color=C_ASR, edgecolor=SURF)
            Cx.set_yticks(y)
            Cx.set_yticklabels(R.quantite, fontsize=8.6, color=INK)
            Cx.set_xlabel("cellules couplees au succes (q < 0,05)",
                          color=INK2, fontsize=9)
            Cx.set_xlim(0, max(4, float(cnt.max()) * 1.25))
            for yy, v in zip(y, cnt.values):
                Cx.text(v + 0.2, yy, f"{int(v)}", va="center", fontsize=8.2,
                        color=INK2)
        Cx.set_title("(c)  detourne contre non detourne", loc="left",
                     fontsize=10, color=INK)
    fig.suptitle("Lectures ancrees sur la derniere ligne d'attention  —  "
                 "pilote, sans entrainement", x=0.010, ha="left",
                 fontsize=12, color=INK)
    fig.text(0.010, 0.006,
             "Wilcoxon apparie bilateral (a, b) et Mann-Whitney bilateral (c), "
             "Benjamini-Hochberg q<0,05 par famille. Aucune regression "
             "logistique, aucun seuil, aucune AUROC.",
             fontsize=7.8, color=MUTED, ha="left")
    fig.subplots_adjust(left=0.085, right=0.985, top=0.92, bottom=0.135)
    fig.savefig(out_png, dpi=160, facecolor=SURF)
    plt.close(fig)
    return out_png


def main() -> int:
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass
    racine = sys.argv[1] if len(sys.argv) > 1 else RACINE
    tags = [t for t in sorted(os.listdir(racine))
            if os.path.isfile(os.path.join(racine, t, "scalaires.parquet"))]
    res, sens = {}, {}
    for tag in tags:
        dd = os.path.join(racine, tag)
        prov = json.load(open(os.path.join(dd, "provenance.json")))
        D = pd.read_parquet(os.path.join(dd, "scalaires.parquet"))
        paires = sorted(D.pair_id.unique())
        Da, Db = delta(D, ARM_ADV), delta(D, ARM_BEN)
        Ta, Tb = tests(Da, paires), tests(Db, paires)
        R = resume(Ta, Tb)
        Tasr, comp = couplage_asr(Da, tag)
        Ta.to_csv(os.path.join(dd, "tests_injection.csv"), index=False)
        Tb.to_csv(os.path.join(dd, "tests_benin.csv"), index=False)
        R.to_csv(os.path.join(dd, "resume.csv"), index=False)
        if Tasr is not None:
            Tasr.to_csv(os.path.join(dd, "couplage_succes.csv"), index=False)
        res[tag] = (R, Ta, Tb, Tasr, int(prov["n_layers"]))
        sens[tag] = R.set_index("quantite").part_pos_bh

        print("=" * 100)
        print(f"{tag}  --  {prov['n_pairs_kept']} paires  --  "
              f"{prov['n_layers']} couches x {prov['n_heads']} tetes  --  "
              f"ancre : {prov['ancre']}")
        print(f"\n{'quantite':<12}{'famille':<12}{'BH inj':>8}{'BH ben':>8}"
              f"{'|dz| inj':>10}{'|dz| ben':>10}{'rapp':>7}{'% dz>0':>9}"
              f"{'|dz| max':>10}{'cellule':>9}")
        for _, r in R.iterrows():
            pp = "   --" if not np.isfinite(r.part_pos_bh) else f"{r.part_pos_bh:>8.0%}"
            print(f"{r.quantite:<12}{r.famille:<12}{r.bh_inj:>5}/{r.n_cellules:<3}"
                  f"{r.bh_ben:>7}{r.absdz_inj:>10.2f}{r.absdz_ben:>10.2f}"
                  f"{r.rapport_med:>7.1f}{pp}{r.absdz_max:>10.2f}"
                  f"{r.cell_max:>9}")
        print("   % dz>0 = part des cellules retenues ou l'injection AUGMENTE "
              "la quantite (coherence de direction)")
        print("   cellule = celle de |dz| maximal : DESCRIPTIVE, choisie apres "
              "resultats, ne pas l'utiliser comme cellule declaree")
        if Tasr is not None:
            cnt = Tasr.groupby("quantite").signif.sum().reindex(R.quantite)
            print(f"\n-- couplage au succes (bras C4, paires a ASR defini)")
            print(comp.to_string())
            print("   cellules couplees au succes, q<0,05 : "
                  + ", ".join(f"{q}={int(v)}" for q, v in cnt.items()))
    if len(sens) == 2:
        (t1, s1), (t2, s2) = sens.items()
        print("=" * 100)
        print(f"-- concordance du SENS majoritaire des cellules retenues, "
              f"{t1} contre {t2}")
        for q in QUANTITES:
            a, b = s1.get(q, np.nan), s2.get(q, np.nan)
            if not (np.isfinite(a) and np.isfinite(b)):
                print(f"   {q:<12} indefini (aucune cellule retenue d'un cote)")
                continue
            sa, sb = np.sign(a - 0.5), np.sign(b - 0.5)
            print(f"   {q:<12} {a:>5.0%}  {b:>5.0%}   "
                  f"{'MEME SENS' if sa == sb and sa != 0 else 'sens oppose ou neutre'}")
    out = os.path.join(racine, "lectures_ancrees.png")
    print(f"\nfigure : {figure(res, out)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
