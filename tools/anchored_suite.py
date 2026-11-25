"""Suite d'experiences sur les lectures ancrees. CPU, sans entrainement.

Prolonge tools/anchored_readout.py (le pilote) sur trois axes, mesures dans
une seule passe par bras pour que tout soit apparie.

1. QUATRIEME BRAS, ``C3_align`` -- separer << instruction >> et << conflit >>.
   Dans OPI, l'instruction injectee I_inj est la consigne OFFICIELLE d'une
   autre tache (une formulation par tache). Le bras C3_align met dans le slot
   la consigne officielle de la MEME tache que la tache legitime : meme
   registre, adressee au modele, mais qui ne detourne rien. Aucun texte n'est
   ecrit pour l'experience : la consigne vient du dataset.
   Appariement exact au token, comme les autres remplissages : si la consigne
   est trop longue, coupe au dernier mot qui tient ; s'il manque des tokens,
   complement en prose benigne du dataset (vivier de BenignFiller, memes
   exclusions). La part du slot occupee par la consigne est ecrite par paire.
   Decomposition EXACTE, puisque les deux sont relatifs a C3_opi :
       (C4 - C3_opi) = (C3_align - C3_opi) + (C4 - C3_align)
          injection  =    forme d'instruction  +     conflit
2. DEUX ANCRES.
   ``fin``   dernier token du prompt (ancre du pilote)
   ``finE``  dernier token du bloc externe E. C'est le MEME token dans les
             quatre bras (fin des donnees injectees, texte identique) :
             seule change ce qui est en amont, dans le slot.
3. DEUX PONDERATIONS.
   ``attn``  poids d'attention bruts
   ``vw``    aretes ponderees par les valeurs (spectral_trust 0.5.3,
             value_weighted_attention), ligne renormalisee a 1. Le puits
             pese beaucoup moins dans ce graphe : la question est de savoir
             si l'effet survit.

Quantites : celles du pilote. A l'ancre ``finE``, les quantites << mbar >>
(ligne moyenne du suffixe de template) n'ont pas de sens et valent NaN.

AUCUNE regression, aucun seuil, aucune AUROC. Le puits n'est ni retire ni
renormalise : a0 est une mesure.

    python tools/anchored_suite.py --model meta-llama/Llama-3.2-1B-Instruct
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

from anchored_readout import E_REGIONS, QUANTITES, scalaires   # noqa: E402

ARM_REF, ARM_ADV, ARM_BEN, ARM_ALN = "C3_opi", "C4", "C3_wiki", "C3_align"
ARMS = (ARM_REF, ARM_ADV, ARM_BEN, ARM_ALN)
ANCRES = ("fin", "finE")
MAX_CANDIDATS = 400


class AlignedFiller:
    """Consigne officielle de la tache LEGITIME, ajustee au token pres."""

    def __init__(self, df, seed=0):
        from spectral_injection.opi import dataset as ds
        self.consigne = df.groupby("injected_task").I_inj.first().to_dict()
        self.pool = df.normal_input.drop_duplicates().tolist()
        self.seed = seed
        self._grams = ds._grams
        self._words = ds._WORDS

    def _coupe(self, txt, n_tok, prefix, suffix, base, cible):
        """Plus long prefixe au mot de ``txt`` donnant <= cible tokens."""
        ends = [m.end() for m in self._words.finditer(txt)]
        lo, hi, best = 0, len(ends) - 1, -1
        while lo <= hi:
            mid = (lo + hi) // 2
            d = n_tok(prefix + txt[:ends[mid]] + " " + suffix) - base
            if d <= cible:
                best, lo = mid, mid + 1
            else:
                hi = mid - 1
        return (txt[:ends[best]] + " ") if best >= 0 else ""

    def fit(self, row, n_tokens, target):
        prefix = row.normal_input + row.P_atk
        suffix = row.D_inj
        n_tok = lambda s: n_tokens(row.instruction, s)          # noqa: E731
        base = n_tok(prefix + suffix)
        cons = self.consigne[row.task_type]
        d = n_tok(prefix + cons + suffix) - base
        if d == target:
            return cons, {"kind": "exact", "part_consigne": 1.0}
        if d > target:
            cons = self._coupe(cons, n_tok, prefix, suffix, base, target)
            if not cons:
                return None, {"kind": "echec"}
            d = n_tok(prefix + cons + suffix) - base
            if d == target:
                return cons, {"kind": "coupee", "part_consigne": 1.0}
        reste = target - d
        host = self._grams(row.normal_input)
        rng = np.random.default_rng(self.seed + 7919 + int(row.row_index))
        p2 = prefix + cons
        for j in rng.permutation(len(self.pool))[:MAX_CANDIDATS]:
            cand = self.pool[j]
            if cand == row.normal_input or (self._grams(cand) & host):
                continue
            pad = self._coupe(cand, n_tok, p2, suffix, d + base - 0, reste)
            if pad and n_tok(p2 + pad + suffix) - base == target:
                return cons + pad, {"kind": "completee",
                                    "part_consigne": round(d / target, 3)}
        return None, {"kind": "echec"}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--pairs", type=int, default=56)
    ap.add_argument("--per-stratum", type=int, default=2)
    ap.add_argument("--seed", type=int, default=2027)
    ap.add_argument("--max-tokens", type=int, default=512)
    ap.add_argument("--st053", default=None)
    ap.add_argument("--no-vw", action="store_true")
    ap.add_argument("--out", default=os.path.join("results", "opi",
                                                  "anchored_suite"))
    args = ap.parse_args()
    if args.st053:
        sys.path.insert(0, os.path.abspath(args.st053))
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

    import torch
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
    log(f"=== suite ancree  {args.model} -> {out_dir}")
    df, _ = ds.load()
    rows = sample(df, args.per_stratum, args.seed).head(args.pairs)
    template, _b = tm.resolve(args.model)
    cap = MultiCapturer(args.model, dtype="float32",
                        max_tokens=args.max_tokens)
    L_layers, H_heads = cap.n_layers, cap.n_heads
    log(f"{L_layers} couches x {H_heads} tetes ; bras {ARMS} ; ancres {ANCRES}")

    VW = None
    if not args.no_vw:
        try:
            from spectral_trust import value_weighted as VW
        except Exception as e:
            log(f"value_weighted indisponible : {type(e).__name__} {e}")

    def n_tokens(instruction, external):
        txt = cm.format_user_only(cap.tokenizer,
                                  cm.build_user_turn(instruction, external),
                                  template)
        return len(cap.tokenizer(txt, add_special_tokens=False)["input_ids"])

    fillers = {"C3_opi": ds.BenignFiller(df, seed=args.seed),
               "C3_wiki": wf.WikitextFiller(seed=args.seed)}
    align = AlignedFiller(df, seed=args.seed)

    recs, meta = [], []
    skipped = {"unmatched": 0, "align": 0, "too_long": 0, "audit": 0,
               "deltaN": 0, "labels": 0}
    vw_ok, vw_msg = VW is not None, ""
    for pi, (_, row) in enumerate(rows.iterrows()):
        pid = (f"{row.attack_type}__{row.task_type}__{row.injected_task}"
               f"__r{int(row.row_index)}")
        arms = build_arms(row, fillers, n_tokens)
        if arms is None:
            skipped["unmatched"] += 1
            continue
        target = arms["target_tokens"]
        txt, diag = align.fit(row, n_tokens, target)
        if txt is None:
            skipped["align"] += 1
            continue
        arms["texts"][ARM_ALN] = ds.external_C3(row, txt)
        arms["comps"][ARM_ALN] = r6.components(row, txt)
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
        if not ok or not all(r6.audit(labels[ARM_REF], labels[a])["ok"]
                             for a in (ARM_ADV, ARM_ALN)):
            skipped["audit"] += 1
            continue
        if len({len(segs[a]) for a in ARMS}) != 1:
            skipped["deltaN"] += 1
            continue
        if not all(np.array_equal(labels[ARM_REF], labels[a])
                   for a in ARMS[1:]):
            skipped["labels"] += 1
            continue

        lab = labels[ARM_REF]
        T = lab.size
        i_task = np.flatnonzero(lab == "task")
        i_E = np.flatnonzero(np.isin(lab, E_REGIONS))
        i_slot = np.flatnonzero(lab == "slot")
        contenu = np.flatnonzero(lab != "template")
        i_suf = np.arange(contenu.max() + 1, T)
        if (i_suf.size == 0 or i_slot.size == 0 or i_task.size == 0
                or lab[0] != "template"):
            skipped["labels"] += 1
            continue
        t_finE = int(i_E.max())
        t_temoin = int(i_task.max())
        lignes = {"fin": T - 1, "finE": t_finE}

        tp = time.time()
        temoin = {}
        for arm in ARMS:
            att = cap.forward(segs[arm], check=True, with_hidden=False)
            att = att[0] if isinstance(att, tuple) else att
            vw = None
            if vw_ok:
                try:
                    vw = VW.value_weighted_attention(
                        cap.model, list(segs[arm].input_ids), per_head=True,
                        tokenizer=cap.tokenizer)
                    if sorted(vw) != list(range(L_layers)):
                        raise ValueError(f"couches vw {sorted(vw)[:4]}...")
                except Exception as e:
                    vw_ok, vw_msg, vw = False, f"{type(e).__name__}: {e}", None
                    log(f"   ponderation par les valeurs DESACTIVEE : {vw_msg[:160]}")
            for l in range(L_layers):
                A = np.asarray(att[l], dtype=np.float64)
                temoin[(arm, l)] = A[:, t_temoin, :t_temoin + 1].copy()
                graphes = {"attn": A}
                if vw is not None:
                    V = vw[l]
                    V = (V[0] if V.dim() == 4 else V).to(torch.float64).numpy()
                    graphes["vw"] = V
                for poids, G in graphes.items():
                    for ancre, t in lignes.items():
                        R = G[:, t, :].copy()
                        s = R.sum(axis=1, keepdims=True)
                        R = R / np.where(s > 0, s, 1.0)
                        # matrice dont la derniere ligne est la ligne ancree,
                        # et dont les lignes du suffixe servent a mbar
                        Ga = G / np.where(G.sum(axis=2, keepdims=True) > 0,
                                          G.sum(axis=2, keepdims=True), 1.0)
                        Ga = Ga.copy()
                        Ga[:, -1, :] = R
                        q = scalaires(Ga, i_task, i_E, i_slot, i_suf)
                        if ancre == "finE":
                            q["mbar_task"] = np.full(H_heads, np.nan)
                            q["mbar_slot"] = np.full(H_heads, np.nan)
                        for h in range(H_heads):
                            r = {"pair_id": pid, "arm": arm, "ancre": ancre,
                                 "poids": poids, "couche": l, "tete": h}
                            r.update({k: float(q[k][h]) for k in QUANTITES})
                            recs.append(r)
            del att, vw
        dt = max(float(np.abs(temoin[(a, l)] - temoin[(ARM_REF, l)]).max())
                 for a in ARMS[1:] for l in range(L_layers))
        meta.append({"pair_id": pid, "T": T, "attack_type": row.attack_type,
                     "task_type": row.task_type,
                     "injected_task": row.injected_task,
                     "n_slot": int(i_slot.size),
                     "align_kind": diag["kind"],
                     "part_consigne": diag.get("part_consigne", np.nan),
                     "temoin_max_ecart": dt})
        log(f"[{pi + 1}/{len(rows)}] {pid}  T={T} slot={i_slot.size} "
            f"align={diag['kind']}({diag.get('part_consigne', 0):.2f}) "
            f"temoin {dt:.1e}  {time.time() - tp:.1f}s")

    D = pd.DataFrame(recs)
    M = pd.DataFrame(meta)
    D.to_parquet(os.path.join(out_dir, "scalaires.parquet"), index=False)
    M.to_csv(os.path.join(out_dir, "paires.csv"), index=False)
    log("")
    log(f"{len(M)} paires retenues ; ignorees {skipped}")
    log(f"controle temoin : ecart max {M.temoin_max_ecart.max():.3e} (attendu 0)")
    log(f"bras C3_align : {M.align_kind.value_counts().to_dict()} ; part "
        f"mediane de la consigne dans le slot {M.part_consigne.median():.2f}")
    log(f"ponderation par les valeurs : "
        f"{'OK' if vw_ok else 'NON (' + vw_msg[:120] + ')'}")
    json.dump({"model": args.model, "tag": args.tag, "n_layers": L_layers,
               "n_heads": H_heads, "arms": list(ARMS), "ancres": list(ANCRES),
               "poids": ["attn"] + (["vw"] if vw_ok else []),
               "vw_message": vw_msg, "n_pairs_kept": len(M),
               "skipped": skipped, "seed": args.seed,
               "python": platform.python_version(),
               "secondes": round(time.time() - t0, 1)},
              open(os.path.join(out_dir, "provenance.json"), "w"), indent=2)
    log(f"termine en {time.time() - t0:.0f}s")
    logf.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
