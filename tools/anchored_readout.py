"""Pilote : lectures ANCREES sur la derniere ligne d'attention, sans entrainement.

ORIGINE. Transposition des << anchored readouts >> (Identity-Preserving
Attention Readouts for Deductive Validity, soumission ICLR 2027) a
l'injection de prompt. Le papier lit six scalaires sur la ligne du dernier
token de la conclusion, puis ajuste une regression logistique sur les
labels. ICI, AUCUNE REGRESSION : les scalaires sont mesures et testes un par
un, avec le protocole du depot. C'est une caracterisation, pas un detecteur.

L'ANCRE. Le dernier token du prompt, fin de l'en-tete assistant : la
position d'ou part la generation, analogue du dernier token de la
conclusion.

QUANTITES DECLAREES AVANT MESURE, par (couche, tete). Notations : a est la
ligne du dernier token, a_bar la ligne MOYENNE du suffixe de template (les
tokens de format qui suivent le contenu utilisateur), E le bloc externe
= normal + preamble + slot + tail.

  famille DEPLOYABLE -- n'utilise que les frontieres de E, que
  l'application connait puisque c'est elle qui insere les donnees externes
    m_task      sum_{j in task} a_j        masse vers l'instruction legitime
    m_E         sum_{j in E} a_j           masse vers les donnees externes
    a0          a_0                        masse vers le puits (BOS)
    H           -sum_j a_j log a_j         entropie de la ligne
    amax        max_j a_j                  maximum de la ligne
    mbar_task   sum_{j in task} a_bar_j    masse task de la ligne moyenne
  H et amax sont symetriques en les cles : ils n'utilisent AUCUNE etiquette.

  famille ORACLE -- utilise la position exacte de l'injection, qu'un
  detecteur deploye n'a pas. Rendue pour comparaison, jamais comme resultat
  de detection.
    m_slot      sum_{j in slot} a_j
    mbar_slot   sum_{j in slot} a_bar_j

Les deux familles sont attention-seule. Elles sont separees parce que
l'une est deployable et l'autre non ; ce decoupage est un choix declare.

CE QUI N'EST PAS FAIT, ET POURQUOI.
  - Pas de regression logistique ni de combinaison apprise : contrainte.
  - Pas d'AUROC : elle exige de choisir un sens, donc de presupposer la
    direction de l'effet. Decision laissee a l'utilisateur.
  - Pas de plancher de surface TF-IDF : c'est un classifieur ajuste sur
    les labels. Il faut une confirmation explicite avant de l'ajouter.
  - Aucun retrait du puits : a0 est une MESURE de la masse que le puits
    recoit ; la ligne n'est ni tronquee ni renormalisee.

CONTROLES.
  1. Ancre TEMOIN au dernier token de ``task`` : par causalite, sa ligne
     est bit-identique entre les trois bras. Ecart maximal attendu 0.
  2. Chaque ligne somme a 1 ; m_task + m_E + m_template = 1.
  3. La position 0 est etiquetee ``template`` (le BOS).

    python tools/anchored_readout.py --model meta-llama/Llama-3.2-1B-Instruct
        --tag llama3.2-1b --pairs 56 --st053 <dossier 0.5.3>
"""

import argparse
import json
import os
import platform
import sys
import time

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

ARM_REF, ARM_ADV, ARM_BEN = "C3_opi", "C4", "C3_wiki"
ARMS = (ARM_REF, ARM_ADV, ARM_BEN)
E_REGIONS = ("normal", "preamble", "slot", "tail")

FAM = {"deployable": ["m_task", "m_E", "a0", "H", "amax", "mbar_task"],
       "oracle": ["m_slot", "mbar_slot"]}
QUANTITES = [q for v in FAM.values() for q in v]


def scalaires(A_l, i_task, i_E, i_slot, i_suf):
    """A_l : [H, T, T]. -> dict quantite -> [H]."""
    a = A_l[:, -1, :]                                   # [H, T]
    abar = A_l[:, i_suf, :].mean(axis=1)                # [H, T]
    with np.errstate(divide="ignore", invalid="ignore"):
        ent = -np.where(a > 0, a * np.log(a), 0.0).sum(axis=1)
    return {
        "m_task": a[:, i_task].sum(axis=1),
        "m_E": a[:, i_E].sum(axis=1),
        "a0": a[:, 0],
        "H": ent,
        "amax": a.max(axis=1),
        "mbar_task": abar[:, i_task].sum(axis=1),
        "m_slot": a[:, i_slot].sum(axis=1),
        "mbar_slot": abar[:, i_slot].sum(axis=1),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--pairs", type=int, default=56)
    ap.add_argument("--per-stratum", type=int, default=2)
    ap.add_argument("--seed", type=int, default=2027)
    ap.add_argument("--max-tokens", type=int, default=512)
    ap.add_argument("--st053", default=None)
    ap.add_argument("--out", default=os.path.join("results", "opi", "anchored"))
    args = ap.parse_args()
    if args.st053:
        sys.path.insert(0, os.path.abspath(args.st053))
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

    from spectral_injection.opi import capture as cm
    from spectral_injection.opi import dataset as ds
    from spectral_injection.opi import regions6 as r6
    from spectral_injection.opi import templates as tm
    from spectral_injection.opi import wikifiller as wf
    from run_region_energy import MultiCapturer, build_arms, sample

    out_dir = os.path.join(args.out, args.tag)
    os.makedirs(out_dir, exist_ok=True)
    logf = open(os.path.join(out_dir, "run.log"), "w", encoding="utf-8",
                buffering=1)

    def log(*x):
        s = " ".join(str(v) for v in x)
        print(s, flush=True)
        logf.write(s + "\n")

    t0 = time.time()
    log(f"=== lectures ancrees (pilote)  {args.model} -> {out_dir}")
    log(f"familles : " + " | ".join(f"{k}={v}" for k, v in FAM.items()))

    df, _ = ds.load()
    rows = sample(df, args.per_stratum, args.seed).head(args.pairs)
    template, _b = tm.resolve(args.model)
    cap = MultiCapturer(args.model, dtype="float32",
                        max_tokens=args.max_tokens)
    L_layers, H_heads = cap.n_layers, cap.n_heads
    log(f"{L_layers} couches x {H_heads} tetes ; ancre = dernier token")

    def n_tokens(instruction, external):
        txt = cm.format_user_only(cap.tokenizer,
                                  cm.build_user_turn(instruction, external),
                                  template)
        return len(cap.tokenizer(txt, add_special_tokens=False)["input_ids"])

    fillers = {"C3_opi": ds.BenignFiller(df, seed=args.seed),
               "C3_wiki": wf.WikitextFiller(seed=args.seed)}

    recs, meta, ctrl = [], [], []
    skipped = {"unmatched": 0, "too_long": 0, "audit": 0, "deltaN": 0,
               "labels": 0}
    for pi, (_, row) in enumerate(rows.iterrows()):
        pid = (f"{row.attack_type}__{row.task_type}__{row.injected_task}"
               f"__r{int(row.row_index)}")
        arms = build_arms(row, fillers, n_tokens)
        if arms is None:
            skipped["unmatched"] += 1
            continue
        if n_tokens(row.instruction, arms["texts"][ARM_ADV]) > args.max_tokens:
            skipped["too_long"] += 1
            continue
        segs, labels, ok = {}, {}, True
        for arm in ARMS:
            seg = cm.segment(cap.tokenizer, row.instruction,
                             arms["texts"][arm], template)
            lab = r6.label_tokens(seg, row.instruction, arms["comps"][arm])
            if not np.array_equal(r6.to_three(lab), seg.regions):
                ok = False
                break
            segs[arm], labels[arm] = seg, np.asarray(lab)
        if not ok or not r6.audit(labels[ARM_REF], labels[ARM_ADV])["ok"]:
            skipped["audit"] += 1
            continue
        if len({len(segs[a]) for a in ARMS}) != 1:
            skipped["deltaN"] += 1
            continue
        if not all(np.array_equal(labels[ARM_REF], labels[a])
                   for a in (ARM_ADV, ARM_BEN)):
            skipped["labels"] += 1
            continue

        lab = labels[ARM_REF]
        T = lab.size
        i_task = np.flatnonzero(lab == "task")
        i_E = np.flatnonzero(np.isin(lab, E_REGIONS))
        i_slot = np.flatnonzero(lab == "slot")
        i_tpl = np.flatnonzero(lab == "template")
        contenu = np.flatnonzero(lab != "template")
        i_suf = np.arange(contenu.max() + 1, T)         # suffixe de template
        if (i_suf.size == 0 or i_slot.size == 0 or i_task.size == 0
                or lab[0] != "template" or lab[-1] != "template"):
            skipped["labels"] += 1
            continue
        t_temoin = int(i_task.max())

        tp = time.time()
        temoin = {}
        for arm in ARMS:
            att = cap.forward(segs[arm], check=True, with_hidden=False)
            att = att[0] if isinstance(att, tuple) else att
            for l in range(L_layers):
                A = np.asarray(att[l], dtype=np.float64)    # [H, T, T]
                s = scalaires(A, i_task, i_E, i_slot, i_suf)
                somme = A[:, -1, :].sum(axis=1)
                part = (s["m_task"] + s["m_E"]
                        + A[:, -1, i_tpl].sum(axis=1))
                ctrl.append({"pair_id": pid, "arm": arm, "couche": l,
                             "max_ecart_somme": float(np.abs(somme - 1).max()),
                             "max_ecart_partition": float(
                                 np.abs(part - somme).max())})
                temoin[(arm, l)] = A[:, t_temoin, :t_temoin + 1].copy()
                for h in range(H_heads):
                    r = {"pair_id": pid, "arm": arm, "couche": l, "tete": h}
                    r.update({q: float(s[q][h]) for q in QUANTITES})
                    recs.append(r)
            del att
        dt = max(float(np.abs(temoin[(a, l)] - temoin[(ARM_REF, l)]).max())
                 for a in (ARM_ADV, ARM_BEN) for l in range(L_layers))
        meta.append({"pair_id": pid, "T": T, "attack_type": row.attack_type,
                     "task_type": row.task_type,
                     "injected_task": row.injected_task,
                     "n_task": int(i_task.size), "n_E": int(i_E.size),
                     "n_slot": int(i_slot.size), "n_suffixe": int(i_suf.size),
                     "temoin_max_ecart": dt})
        log(f"[{pi + 1}/{len(rows)}] {pid}  T={T} task={i_task.size} "
            f"E={i_E.size} slot={i_slot.size} suffixe={i_suf.size}  "
            f"temoin {dt:.1e}  {time.time() - tp:.1f}s")

    D = pd.DataFrame(recs)
    M = pd.DataFrame(meta)
    C = pd.DataFrame(ctrl)
    D.to_parquet(os.path.join(out_dir, "scalaires.parquet"), index=False)
    M.to_csv(os.path.join(out_dir, "paires.csv"), index=False)
    C.to_csv(os.path.join(out_dir, "controles.csv"), index=False)
    log("")
    log(f"{len(M)} paires retenues ; ignorees {skipped}")
    log(f"controle 1, ancre temoin (dernier token de task) : ecart max "
        f"{M.temoin_max_ecart.max():.3e}  (attendu 0)")
    log(f"controle 2, somme des lignes : ecart max "
        f"{C.max_ecart_somme.max():.3e} ; partition task+E+template : "
        f"{C.max_ecart_partition.max():.3e}")
    json.dump({"model": args.model, "tag": args.tag,
               "n_layers": L_layers, "n_heads": H_heads,
               "familles": FAM, "family_kind": "attention-only",
               "ancre": "dernier token du prompt",
               "arms": list(ARMS), "n_pairs_kept": len(M),
               "skipped": skipped, "seed": args.seed,
               "per_stratum": args.per_stratum,
               "python": platform.python_version(),
               "secondes": round(time.time() - t0, 1)},
              open(os.path.join(out_dir, "provenance.json"), "w"), indent=2)
    log(f"termine en {time.time() - t0:.0f}s")
    logf.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
