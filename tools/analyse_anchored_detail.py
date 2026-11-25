"""Analyses complementaires du pilote des lectures ancrees. Declarees avant
de voir les resultats ; aucune n'entraine quoi que ce soit.

1. REPLICATION PAR MOITIES. Paires triees, rang pair = selection, rang
   impair = confirmation. Pour chaque quantite, les K = 10 cellules de |dz|
   maximal sur la moitie de selection sont relues sur l'autre moitie : part
   de meme signe, part confirmee (p < 0,05 bilateral), |dz| median avant et
   apres. Une chute forte de |dz| signe la malediction du vainqueur.

2. PAR TYPE D'ATTAQUE. Sur les cellules retenues par BH (contraste
   injection, donnees completes), part des (paire, cellule) d'un type
   d'attaque dont Delta va dans le sens majoritaire de la cellule. Dit si
   l'effet existe pour chaque type ou est porte par un seul.

3. UNITES NATURELLES. Sur les cellules retenues, |Delta| median en points
   de masse d'attention (x 100) et rapporte a la valeur de reference.

4. E CONTRE SLOT. Delta m_E = somme des Delta sur normal, preamble, slot,
   tail. Sur les cellules ou m_E est retenue : mediane de
   Delta m_slot / Delta m_E. Proche de 1 : le mouvement de la version
   deployable est porte par l'injection ; loin de 1 : il vient du reste
   du bloc externe.

5. SUCCES, STRATIFIE. Sur les cellules couplees au succes, signe de
   l'ecart detourne - non detourne a l'interieur de chaque type d'attaque
   ou les deux issues existent. Un ecart qui change de signe entre strates
   est un effet de composition.

    python tools/analyse_anchored_detail.py
"""

import json
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from analyse_region_energy import dz, paired_test               # noqa: E402
from analyse_anchored import (ASR_RACINE, FAM, QUANTITES, RACINE,  # noqa: E402
                              delta)
from anchored_readout import ARM_ADV, ARM_REF                    # noqa: E402

K_SEL = 10


def _piv(Dl, q, paires):
    return Dl.pivot_table(index="pair_id", columns=["couche", "tete"],
                          values=q).reindex(paires)


def moities(Dl, paires):
    sel, conf = paires[0::2], paires[1::2]
    rows = []
    for q in QUANTITES:
        Ps, Pc = _piv(Dl, q, sel), _piv(Dl, q, conf)
        ds = dz(Ps.to_numpy(np.float64))
        ok = np.isfinite(ds)
        idx = np.argsort(-np.where(ok, np.abs(ds), -1))[:K_SEL]
        Vc = Pc.to_numpy(np.float64)[:, idx]
        dc = dz(Vc)
        pc = paired_test(Vc)
        rows.append({"quantite": q,
                     "meme_signe": float(np.mean(np.sign(dc) == np.sign(ds[idx]))),
                     "confirme": float(np.mean((pc < 0.05)
                                               & (np.sign(dc) == np.sign(ds[idx])))),
                     "absdz_sel": float(np.median(np.abs(ds[idx]))),
                     "absdz_conf": float(np.median(np.abs(dc)))})
    return pd.DataFrame(rows)


def par_attaque(Dl, Ta):
    Dl = Dl.assign(attaque=Dl.pair_id.str.split("__").str[0])
    rows = []
    for q in QUANTITES:
        sig = Ta[(Ta.quantite == q) & Ta.signif]
        if sig.empty:
            continue
        sens = dict(zip(zip(sig.couche, sig.tete), np.sign(sig.dz)))
        d = Dl[[ (c, h) in sens for c, h in zip(Dl.couche, Dl.tete)]].copy()
        d["s"] = [sens[(c, h)] for c, h in zip(d.couche, d.tete)]
        d["ok"] = np.sign(d[q]) == d.s
        r = {"quantite": q, "n_cellules": len(sig)}
        r.update(d.groupby("attaque").ok.mean().round(3).to_dict())
        rows.append(r)
    return pd.DataFrame(rows)


def unites(D, Dl, Ta):
    k = ["pair_id", "couche", "tete"]
    ref = D[D.arm == ARM_REF].set_index(k)
    rows = []
    for q in QUANTITES:
        sig = Ta[(Ta.quantite == q) & Ta.signif]
        if sig.empty:
            continue
        cel = set(zip(sig.couche, sig.tete))
        m = [(c, h) in cel for c, h in zip(Dl.couche, Dl.tete)]
        d = Dl.loc[m].set_index(k)
        r0 = ref.loc[d.index, q]
        rel = (d[q].abs() / r0.abs().replace(0, np.nan))
        rows.append({"quantite": q, "n_cellules": len(sig),
                     "abs_delta_med": float(d[q].abs().median()),
                     "reference_med": float(r0.median()),
                     "relatif_med": float(rel.median())})
    return pd.DataFrame(rows)


def e_contre_slot(Dl, Ta):
    sig = Ta[(Ta.quantite == "m_E") & Ta.signif]
    if sig.empty:
        return None
    cel = set(zip(sig.couche, sig.tete))
    m = [(c, h) in cel for c, h in zip(Dl.couche, Dl.tete)]
    d = Dl.loc[m]
    d = d[d.m_E.abs() > 1e-6]
    r = d.m_slot / d.m_E
    return {"n_cellules": len(sig), "rapport_med": float(r.median()),
            "q25": float(r.quantile(0.25)), "q75": float(r.quantile(0.75)),
            "meme_signe": float((np.sign(d.m_slot) == np.sign(d.m_E)).mean())}


def succes_strat(Dl, tag):
    f_c = os.path.join(RACINE, tag, "couplage_succes.csv")
    f_p = os.path.join(ASR_RACINE, tag, "pairs.csv")
    if not (os.path.exists(f_c) and os.path.exists(f_p)):
        return None
    Cs = pd.read_csv(f_c)
    Cs = Cs[Cs.signif]
    if Cs.empty:
        return pd.DataFrame()
    P = pd.read_csv(f_p)
    P = P[P.asr_defined_C4.fillna(False).astype(bool)]
    hij = dict(zip(P.pair_id, P.asr_hijacked_C4.astype(bool)))
    d = Dl[Dl.pair_id.isin(hij)].assign(
        hij=lambda x: x.pair_id.map(hij),
        attaque=lambda x: x.pair_id.str.split("__").str[0])
    rows = []
    for _, r in Cs.iterrows():
        g = d[(d.couche == r.couche) & (d.tete == r.tete)]
        s_glob = np.sign(r.ecart_med)
        acc, tot = 0, 0
        for a, ga in g.groupby("attaque"):
            x = ga.loc[ga.hij, r.quantite]
            y = ga.loc[~ga.hij, r.quantite]
            if len(x) and len(y):
                tot += 1
                acc += int(np.sign(x.median() - y.median()) == s_glob)
        rows.append({"quantite": r.quantite, "cellule": f"L{r.couche}H{r.tete}",
                     "ecart_global": r.ecart_med, "q_bh": r.q_bh,
                     "strates_meme_signe": acc, "strates_testables": tot})
    return pd.DataFrame(rows)


def main() -> int:
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass
    racine = sys.argv[1] if len(sys.argv) > 1 else RACINE
    for tag in sorted(os.listdir(racine)):
        dd = os.path.join(racine, tag)
        if not os.path.isfile(os.path.join(dd, "tests_injection.csv")):
            continue
        D = pd.read_parquet(os.path.join(dd, "scalaires.parquet"))
        Ta = pd.read_csv(os.path.join(dd, "tests_injection.csv"))
        paires = sorted(D.pair_id.unique())
        Dl = delta(D, ARM_ADV)
        print("=" * 96)
        print(f"{tag}  --  analyses complementaires")
        M = moities(Dl, paires)
        print(f"\n-- 1. replication par moities ({len(paires[0::2])} selection / "
              f"{len(paires[1::2])} confirmation, K={K_SEL})")
        print(M.to_string(index=False, float_format=lambda v: f"{v:.3f}"))
        A = par_attaque(Dl, Ta)
        print("\n-- 2. part des (paire, cellule retenue) dans le sens majoritaire,"
              " par type d'attaque")
        print(A.to_string(index=False) if len(A) else "   aucune cellule retenue")
        U = unites(D, Dl, Ta)
        print("\n-- 3. amplitudes en unites naturelles, cellules retenues")
        print(U.to_string(index=False, float_format=lambda v: f"{v:.4f}")
              if len(U) else "   aucune")
        E = e_contre_slot(Dl, Ta)
        print("\n-- 4. part du mouvement de m_E portee par m_slot")
        print(f"   {E}" if E else "   m_E : aucune cellule retenue")
        S = succes_strat(Dl, tag)
        print("\n-- 5. couplage au succes, stratifie par type d'attaque")
        if S is None:
            print("   pas de labels ASR")
        elif S.empty:
            print("   aucune cellule couplee au succes apres BH")
        else:
            print(S.to_string(index=False))
        M.to_csv(os.path.join(dd, "moities.csv"), index=False)
        A.to_csv(os.path.join(dd, "par_attaque.csv"), index=False)
        U.to_csv(os.path.join(dd, "unites.csv"), index=False)
        if S is not None and len(S):
            S.to_csv(os.path.join(dd, "succes_strat.csv"), index=False)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
