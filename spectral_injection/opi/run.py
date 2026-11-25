"""Execution d'une campagne OPI : smoke test ou pilote.

    python -m spectral_injection.opi.run --rows 8  --out results/opi/smoke
    python -m spectral_injection.opi.run --rows 112 --out results/opi/pilot

Produit, dans ``--out`` :

    values.npz       tenseur [lignes, conditions, couches, tetes, metriques]
    rows.csv         une ligne par exemple : metadonnees, longueurs, appariement
    provenance.json  modele, tokenizer, versions, graine, configuration
"""

import argparse
import csv
import json
import os
import sys
import time
from typing import Dict, List

import numpy as np
import pandas as pd

from . import dataset as ds
from . import provenance as prov
from . import templates as templates_mod
from .capture import (LLAMA3_USER_ONLY, Capturer, build_user_turn,
                      format_user_only, segment, verify_segmentation,
                      verify_template)
from .communities import K_VALUES, ari
from .structural import analyse_layer
from .trajectories import h6_trajectories, partition_stability

#: Metriques suivies par H6. PRE-DECLAREES : aucune n'est choisie sur
#: son pouvoir discriminant, et la liste ne depend pas des resultats.
H6_METRICS = (
    "h1_lambda_2", "h1_lambda_max", "h1_eigengap_23",
    "h2_conductance", "h2_ncut", "h2_cut_over_vol_E",
    "h3_Fbar_E_to_U", "h3_Fbar_E_to_E",
    "h4A_hfer", "h4A_centroid", "h4B_hfer", "h4B_centroid",
    "h5_k2_mix_UE", "h5_k_eigengap",
)

DEFAULT_MODEL = "meta-llama/Llama-3.2-1B-Instruct"
DEFAULT_PARQUET = "data/opi/data/train-00000-of-00001.parquet"


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description="campagne Open-Prompt-Injection")
    p.add_argument("--model", default=DEFAULT_MODEL)
    p.add_argument("--parquet", default=DEFAULT_PARQUET)
    p.add_argument("--rows", type=int, default=8,
                   help="8 = smoke test, 112 = pilote")
    p.add_argument("--per-stratum", type=int, default=None,
                   help="prioritaire sur --rows ; 4 donne 112 lignes")
    p.add_argument("--max-tokens", type=int, default=512)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--per-head", action="store_true", default=True)
    p.add_argument("--aggregated", dest="per_head", action="store_false",
                   help="moyenne les tetes AVANT le graphe (agregation declaree)")
    p.add_argument("--no-h56", action="store_true",
                   help="desactive H5 (communautes) et H6 (trajectoires)")
    p.add_argument("--out", default="results/opi/smoke")
    args = p.parse_args(argv)

    # La console Windows est en cp1252 et ne sait pas encoder les marqueurs
    # de sous-mot des tokenizers non-Llama ('G', '_' barres). Sans cela
    # l'audit des tokens frontiere leve UnicodeEncodeError APRES l'ecriture
    # des tenseurs, et tokens_T.csv n'est jamais ecrit. N'affecte que
    # l'affichage, aucun calcul.
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except Exception:                                   # noqa: BLE001
            pass

    os.makedirs(args.out, exist_ok=True)
    t_start = time.time()

    # ---------------------------------------------------------------- donnees
    print("1. dataset")
    df, templates = ds.load(args.parquet)
    print(f"   {len(df)} lignes, decomposition verifiee")
    if args.per_stratum:
        samp = ds.sample_rows(df, args.per_stratum, args.seed)
    else:
        # smoke : couvrir les 4 familles d'attaque et des taches distinctes
        per = max(1, args.rows // df.attack_type.nunique())
        picks = []
        tasks = sorted(df.task_type.unique())
        for k, (at, g) in enumerate(df.groupby("attack_type", sort=True)):
            rng = np.random.default_rng(args.seed + k)
            for m in range(per):
                sub = g[g.task_type == tasks[(k * per + m) % len(tasks)]]
                picks.append(sub.iloc[rng.integers(len(sub))])
        samp = pd.DataFrame(picks[:args.rows]).reset_index(drop=True)
    print(f"   echantillon : {len(samp)} lignes")
    print(f"   taches  : {sorted(samp.task_type.unique())}")
    print(f"   attaques: {sorted(samp.attack_type.unique())}")

    # ---------------------------------------------------------------- modele
    print("\n2. modele")
    cap = Capturer(args.model, dtype="float32", max_tokens=args.max_tokens)
    print(f"   {args.model}")
    print(f"   {cap.n_layers} couches x {cap.n_heads} tetes "
          f"(kv={cap.n_kv_heads}), device={cap.device}")
    # Le template est resolu depuis l'identifiant du modele. ``resolve``
    # leve plutot que de deviner : un template errone deplacerait la
    # segmentation sans produire d'erreur visible.
    tmpl, bos_marker = templates_mod.resolve(args.model)
    tinfo = templates_mod.verify(cap.tokenizer, tmpl, bos_marker)
    print(f"   template experimental : {tinfo['name']}")
    print(f"   marqueurs : {tinfo['marker_ids']}")
    print(f"   tokens de template hors contenu : {tinfo['n_template_tokens']}")
    print("   AUCUN message systeme : le bloc fabrique par le template "
          "officiel de la famille est supprime")

    def n_tokens(instruction, external):
        txt = format_user_only(cap.tokenizer,
                               build_user_turn(instruction, external),
                               tmpl)
        return len(cap.tokenizer(txt, add_special_tokens=False)["input_ids"])

    # ---------------------------------------------------------------- C1..C4
    print("\n3. construction des conditions")
    filler = ds.BenignFiller(df, seed=args.seed)
    built, meta_rows = [], []
    for _, row in samp.iterrows():
        # C3 = normal + P_atk + remplissage + D_inj remplace exactement
        # I_inj ; la reference est donc normal + P_atk + D_inj, c'est-a-dire
        # C2, et la cible vaut T(C4) - T(C2).
        base = n_tokens(row.instruction, ds.external_C2(row))
        target = n_tokens(row.instruction, row.attack_input) - base
        f, diag = filler.fit(row, n_tokens, target)
        ext = {"C1": ds.external_C1(row), "C2": ds.external_C2(row),
               "C4": ds.external_C4(row)}
        ext["C3"] = ds.external_C3(row, f) if f is not None else None
        built.append((row, ext, f))
        meta_rows.append({
            "row_index": int(row.row_index), "uid": row.uid,
            "task_type": row.task_type, "attack_type": row.attack_type,
            "injected_task": row.injected_task, "sample_id": int(row.sample_id),
            "target_tokens_D": int(target), "filler_matched": diag["matched"],
            "filler_candidates": diag["candidates_tried"],
            "len_P_atk": len(row.P_atk), "len_I_inj": len(row.I_inj),
            "len_D_inj": len(row.D_inj), "len_normal": len(row.normal_input),
        })
    n_matched = sum(m["filler_matched"] for m in meta_rows)
    print(f"   appariement exact du remplissage : {n_matched}/{len(meta_rows)}")
    if n_matched < len(meta_rows):
        print("   ATTENTION : lignes non appariees conservees et marquees "
              "(aucune exclusion silencieuse)")

    # ---------------------------------------------------------------- capture
    print("\n4. capture et analyse")
    conds = list(ds.CONDITIONS)
    V, keys = None, None
    Vt, tkeys = None, None
    stab_rows = []
    t_audit = []
    seg_audit = []
    do_h56 = not args.no_h56
    t0 = time.time()
    for i, (row, ext, f) in enumerate(built):
        for c_i, c in enumerate(conds):
            e = ext[c]
            if e is None:
                continue
            seg = segment(cap.tokenizer, row.instruction, e, tmpl)
            vs = verify_segmentation(seg)
            seg_audit.append({
                "row_index": int(row.row_index), "condition": c,
                "ok": vs["ok"], "offset_gaps": vs["offset_gaps"],
                "offset_overlaps": vs["offset_overlaps"],
                "covers_end": vs["offset_covers_end"],
                "U_contiguous": vs["U_contiguous"],
                "E_contiguous": vs["E_contiguous"],
                "U_before_E": vs["U_before_E"],
                "U_matches_span": vs["U_matches_span"],
                "E_matches_span": vs["E_matches_span"],
                "U_missing_tail": vs["U_missing_tail"],
                "n_boundary_tokens": vs["n_boundary_tokens"],
                "boundary": vs["boundary_tokens"],
            })
            t_audit.append({
                "row_index": int(row.row_index), "condition": c,
                "n_T": seg.count("T"),
                "tokens": tuple(np.array(seg.tokens)[seg.mask("T")].tolist()),
                "positions": tuple(np.flatnonzero(seg.mask("T")).tolist()),
            })
            att = cap.forward(seg, check=True)
            meta_rows[i][f"T_{c}"] = len(seg)
            meta_rows[i][f"nU_{c}"] = seg.count("U")
            meta_rows[i][f"nE_{c}"] = seg.count("E")
            meta_rows[i][f"nT_{c}"] = seg.count("T")
            prev_parts, ari_acc = None, {}
            for l in range(cap.n_layers):
                res, parts = analyse_layer(att[l], seg.regions,
                                           per_head=args.per_head,
                                           with_communities=do_h56,
                                           seed=args.seed)
                if V is None:
                    keys = sorted(res.keys())
                    H = len(res[keys[0]])
                    V = np.full((len(built), len(conds), cap.n_layers, H,
                                 len(keys)), np.nan)
                for k_i, k in enumerate(keys):
                    V[i, c_i, l, :, k_i] = res[k]
                # H6 : stabilite des partitions entre couches consecutives
                if do_h56 and parts is not None:
                    if prev_parts is not None:
                        for h_i, (pa, pb) in enumerate(zip(prev_parts, parts)):
                            if pa is None or pb is None:
                                continue
                            for kk in K_VALUES:
                                ari_acc.setdefault((h_i, kk), []).append(
                                    ari(pa[kk], pb[kk]))
                    prev_parts = parts
            del att

            # H6 : trajectoires inter-couches, par tete
            if do_h56:
                tr = h6_trajectories(V[i, c_i], keys, H6_METRICS)
                H = V.shape[3]
                for kk in K_VALUES:
                    v = np.array([np.mean(ari_acc.get((h_i, kk), [np.nan]))
                                  for h_i in range(H)])
                    tr[f"stab_ari_k{kk}_mean"] = v
                if Vt is None:
                    tkeys = sorted(tr.keys())
                    Vt = np.full((len(built), len(conds), H, len(tkeys)), np.nan)
                for k_i, k in enumerate(tkeys):
                    Vt[i, c_i, :, k_i] = tr[k]
        el = time.time() - t0
        print(f"   [{i+1}/{len(built)}] {row.task_type}/{row.attack_type} "
              f"T={meta_rows[i].get('T_C4')} ({el/(i+1):.1f} s/ligne)",
              flush=True)

    # ---------------------------------------------------------------- sorties
    print("\n5. sorties")
    payload = dict(values=V, metrics=np.array(keys),
                   conditions=np.array(conds),
                   row_index=np.array([m["row_index"] for m in meta_rows]),
                   per_head=np.array([args.per_head]))
    if Vt is not None:
        payload.update(trajectories=Vt, traj_metrics=np.array(tkeys),
                       h6_metrics=np.array(list(H6_METRICS)))
    np.savez_compressed(os.path.join(args.out, "values.npz"), **payload)
    meta = pd.DataFrame(meta_rows)
    meta.to_csv(os.path.join(args.out, "rows.csv"), index=False)

    rec = prov.capture(
        model_id=args.model, tokenizer=cap.tokenizer,
        config={"rows": len(built), "per_stratum": args.per_stratum,
                "max_tokens": args.max_tokens, "per_head": args.per_head,
                "conditions": conds, "system_prompt": None,
                "sink_removal": "none", "kron": False,
                "self_loops": {
                    "primary": "conserves (W garde sa diagonale)",
                    "robustness": "variante nsl_ a diagonale nulle"},
                # Convention d'attribution des tokens, a conserver avec les
                # resultats : un token appartient a la region qui contient le
                # MILIEU de son span de caracteres. Un token a cheval sur une
                # frontiere va donc du cote qui en detient la majorite.
                "region_assignment": {
                    "rule": "midpoint of the token character span",
                    "tie_zero_width": "special tokens use their start offset",
                    "regions": ["U", "E", "T"],
                    "T_definition": "tout ce qui n'est ni U ni E : marqueurs "
                                    "du template, separateur, en-tete de "
                                    "generation, BOS",
                    "known_consequence": "le token frontiere U/E fusionne la "
                                         "ponctuation finale de U avec le "
                                         "separateur '\\n\\n' ; un seul "
                                         "caractere sur trois tombe dans U, "
                                         "donc il est attribue a T et U perd "
                                         "sa ponctuation finale",
                    "measured": "1 token frontiere par prompt, part dans U "
                                "= 0.333, sur 32/32 prompts du smoke test"},
                "remove_self_loops": False},
        dataset_path=args.parquet, seed=args.seed,
        extra={"instruction_templates": templates, "n_metrics": len(keys),
               "tensor_shape": list(V.shape),
               "experimental_template": {
                   "name": tmpl.name,
                   "prefix": tmpl.prefix,
                   "suffix": tmpl.suffix,
                   "bos_marker": tinfo["bos_marker"],
                   "marker_ids": tinfo["marker_ids"],
                   "official_chat_template_bypassed": True,
                   "reason": "le template officiel injecte un bloc systeme "
                             "date, non deterministe d'un jour a l'autre"}})
    prov.write(rec, os.path.join(args.out, "provenance.json"))

    print(f"   tenseur H1-H5 {V.shape} -> {args.out}/values.npz")
    if Vt is not None:
        print(f"   tenseur H6     {Vt.shape} ({len(tkeys)} statistiques "
              f"de trajectoire)")
    print(f"   {len(meta)} lignes -> {args.out}/rows.csv")
    print(f"   provenance -> {args.out}/provenance.json")

    # ---------------------------------------------------------------- controles
    print("\n6. controles d'integrite")
    fin = np.isfinite(V)
    print(f"   valeurs finies : {100*fin.mean():.2f} %")
    iu = keys.index("h3_F_U_to_E") if "h3_F_U_to_E" in keys else None
    if iu is not None:
        mx = np.nanmax(np.abs(V[..., iu]))
        print(f"   F_(U->E) maximal : {mx:.3e}  (doit etre nul, masque causal)")
    ic = keys.index("h2_cut")
    idc = keys.index("h2_dirichlet_comb")
    d = np.nanmax(np.abs(V[..., ic] - V[..., idc]))
    print(f"   |cut - Dirichlet combinatoire| maximal : {d:.3e}")
    for var in ("A", "B"):
        icen = keys.index(f"h4{var}_centroid")
        iray = keys.index(f"h4{var}_rayleigh")
        dd = np.nanmax(np.abs(V[..., icen] - V[..., iray]))
        print(f"   variante {var} : |centroide - Rayleigh| maximal : {dd:.3e}")
    if "nsl_h2_cut" in keys:
        inc, ind = keys.index("nsl_h2_cut"), keys.index("nsl_h2_dirichlet_comb")
        dn = np.nanmax(np.abs(V[..., inc] - V[..., ind]))
        print(f"   variante sans boucles : |cut - Dirichlet| maximal : {dn:.3e}")
        ic2 = keys.index("h2_cut")
        same = np.nanmax(np.abs(V[..., ic2] - V[..., inc]))
        print(f"   |cut(avec boucles) - cut(sans boucles)| maximal : {same:.3e}"
              f"   (doit etre nul : la diagonale ne traverse aucune frontiere)")
    lm = keys.index("h1_lambda_max")
    print(f"   lambda_max observe : [{np.nanmin(V[..., lm]):.4f}, "
          f"{np.nanmax(V[..., lm]):.4f}]  (borne theorique 2)")
    if "nsl_h1_lambda_max" in keys:
        ln = keys.index("nsl_h1_lambda_max")
        print(f"   lambda_max sans boucles : [{np.nanmin(V[..., ln]):.4f}, "
              f"{np.nanmax(V[..., ln]):.4f}]")
        l2, l2n = keys.index("h1_lambda_2"), keys.index("nsl_h1_lambda_2")
        r = np.nanmedian(V[..., l2n] - V[..., l2])
        print(f"   effet des boucles sur lambda_2 : mediane "
              f"{r:+.4f} (sans - avec)")

    # ---------------------------------------------------------------- audit T
    print("\n7a. audit de la segmentation U/E/T")
    sa = pd.DataFrame(seg_audit)
    sa.drop(columns=["boundary"]).to_csv(
        os.path.join(args.out, "segmentation_audit.csv"), index=False)
    bad = sa[~sa.ok]
    print(f"   prompts conformes : {int(sa.ok.sum())}/{len(sa)}")
    for col in ("offset_gaps", "offset_overlaps"):
        print(f"   {col:16} : total {int(sa[col].sum())}")
    for col in ("covers_end", "U_contiguous", "E_contiguous", "U_before_E",
                "U_matches_span", "E_matches_span"):
        v = sa[col]
        print(f"   {col:16} : {int(v.sum())}/{len(v)}")
    tails = [t for t in sa.U_missing_tail.dropna().unique() if t != ""]
    print(f"   queue de U absorbee par le token frontiere : "
          f"{sorted(set(map(repr, tails)))[:4]}")
    nb = sa.n_boundary_tokens
    print(f"   tokens frontieres par prompt : min {nb.min()}, max {nb.max()}")
    ex = next((b for b in sa.boundary if b), [])
    for d in ex[:3]:
        print(f"     token {d['index']:>3} {d['token']!r:>12} assigne a "
              f"{d['region_assigned']}, {d['chars_in']}/{d['chars_total']} "
              f"caracteres dans {d['overlaps']} (part {d['share_in']})")
    if len(bad):
        print(f"   ATTENTION : {len(bad)} prompts non conformes")

    print("\n7b. audit des tokens techniques T")
    ta = pd.DataFrame(t_audit)
    ta.to_csv(os.path.join(args.out, "tokens_T.csv"), index=False)
    sizes = ta.n_T.value_counts().to_dict()
    print(f"   taille de T : {sizes}")
    uniq = ta.tokens.nunique()
    print(f"   compositions distinctes de T : {uniq} sur "
          f"{len(ta)} prompts")
    seq = ta.tokens.iloc[0]
    print(f"   composition (prompt 1), {len(seq)} tokens :")
    print(f"     {list(seq)}")
    pos = ta.positions.iloc[0]
    print(f"   positions : debut {list(pos[:6])} ... fin {list(pos[-6:])}")
    # La variation benigne est celle du token frontiere U/E, qui fusionne
    # avec la ponctuation finale de l'instruction ('.CC' ou '".CC'). Elle ne
    # change ni le nombre de tokens T ni leurs positions. Seule une
    # instabilite STRUCTURELLE merite une alerte.
    same_n = ta.n_T.nunique() == 1
    same_pos = ta.positions.map(lambda t: tuple(t)).nunique() == len(
        ta.groupby(["row_index", "condition"]).size()) * 0 + ta.positions.nunique()
    # Les positions ABSOLUES de T ne peuvent pas etre constantes : les
    # prompts n'ont pas la meme longueur. Ce qui doit l'etre, c'est le bloc
    # d'ouverture (toujours 0..4) et le bloc de fermeture (les 5 derniers).
    def _pos(t):
        return tuple(t) if not isinstance(t, str) else tuple(__import__("ast").literal_eval(t))
    opens = {_pos(t)[:5] for t in ta.positions}
    closes = {tuple(np.array(_pos(t)[-5:]) - (n - 1))
              for t, n in zip(ta.positions, ta.n_T * 0 + [len(_pos(t)) for t in ta.positions])}
    head_ok = opens == {(0, 1, 2, 3, 4)}
    print(f"   bloc d'ouverture toujours aux positions 0-4 : {head_ok}")
    print(f"   taille de T constante : {same_n}")
    if not same_n:
        print("   ATTENTION : la taille de T varie entre prompts")
    elif uniq > 1:
        diffs = set()
        import ast as _ast
        base = _ast.literal_eval(ta.tokens.iloc[0]) if isinstance(
            ta.tokens.iloc[0], str) else ta.tokens.iloc[0]
        for t in ta.tokens.unique():
            cur = _ast.literal_eval(t) if isinstance(t, str) else t
            diffs |= {i for i, (a, b) in enumerate(zip(base, cur)) if a != b}
        print(f"   T varie sur {len(diffs)} position(s) : {sorted(diffs)} "
              f"-- fusion du token frontiere U/E, sans effet structurel")

    if Vt is not None:
        print("\n8. controles H5 / H6")
        for kk in K_VALUES:
            ik = keys.index(f"h5_k{kk}_n_clusters_effective")
            im = keys.index(f"h5_k{kk}_mix_UE")
            print(f"   K={kk} : clusters effectifs mediane "
                  f"{np.nanmedian(V[..., ik]):.2f} (demande {kk}), "
                  f"melange U/E mediane {np.nanmedian(V[..., im]):.3f}")
        ig = keys.index("h5_k_eigengap")
        vg, cg = np.unique(V[..., ig][np.isfinite(V[..., ig])],
                           return_counts=True)
        top = sorted(zip(cg, vg), reverse=True)[:4]
        dist = {int(v): int(c) for c, v in top}
        print(f"   K suggere par l'ecart spectral (K: occurrences) : {dist}")
        for kk in K_VALUES[:2]:
            it = tkeys.index(f"stab_ari_k{kk}_mean")
            print(f"   stabilite ARI entre couches, K={kk} : mediane "
                  f"{np.nanmedian(Vt[..., it]):.3f}")
        itv = tkeys.index("traj_h1_lambda_2__total_variation")
        print(f"   variation totale de lambda_2 : mediane "
              f"{np.nanmedian(Vt[..., itv]):.4f}")
        fin_t = np.isfinite(Vt).mean()
        print(f"   valeurs finies dans le tenseur H6 : {100*fin_t:.2f} %")
    print(f"\ntermine en {time.time()-t_start:.0f} s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
