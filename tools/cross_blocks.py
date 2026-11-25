"""Decomposition croisee signal / graphe, BLOC PAR BLOC, et sur deux contrastes.

CE QU'ON MELANGE. Pour un bloc de regions et une cellule (couche, tete), on
evalue la part d'energie de Dirichlet de ce bloc en croisant le signal d'un
bras avec le laplacien d'un autre :

    graphe seul  = E_blk(X_ref, L_alt) - E_blk(X_ref, L_ref)
    signal seul  = E_blk(X_alt, L_ref) - E_blk(X_ref, L_ref)
    total        = E_blk(X_alt, L_alt) - E_blk(X_ref, L_ref)
    interaction  = total - graphe seul - signal seul

DEUX CONTRASTES, meme reference. ``ref`` est toujours C3_opi.
    contraste "C4"       alt = C4        benin -> injecte
    contraste "C3_wiki"  alt = C3_wiki   benin -> autre benin
Le second est le controle : un terme dont l'amplitude est la meme dans les
deux contrastes se produit des qu'on change le texte du slot, injection ou
pas, et n'apprend rien sur l'injection.

NORMALISATION DECLAREE. Les quatre termes sont divises par
``E_tot(X_ref, L_ref)``, l'energie de Dirichlet TOTALE du bras de
reference, la meme pour les quatre. C'est ce qui preserve l'additivite
exacte ``total = graphe + signal + interaction``. Diviser chaque terme par
son propre total la detruirait.

POURQUOI PAS << zero avant slot >>. Le plan propose de verifier que les
blocs entierement anterieurs au slot donnent zero, par causalite. C'EST
FAUX pour la mesure en 'sym', et le verifier ainsi ferait rejeter un
alignement correct. L'attention est causale, donc A[task, task] et X_task
sont bit-identiques entre les bras ; mais la symetrisation rend
W[task, slot] = A[slot, task] / 2 non nul, donc le degre d'un token de
task depend de ce que contient le slot, donc le facteur D^(-1/2) de
L_sym en depend aussi. Un Delta non nul sur task--task est attendu, et
mesure exactement cet effet de renormalisation.

LE CONTROLE CORRECT, implemente ici, est en trois parties :
  1. X_task bit-identique entre les bras        (causalite sur le signal)
  2. A[task, :task] bit-identique entre les bras (causalite sur le graphe)
  3. en normalisation 'none', ou L = D - W n'a aucun facteur de degre,
     le Delta des blocs entierement anterieurs au slot doit etre
     EXACTEMENT nul. C'est la version juste de la verification gratuite.

FAMILLE HYBRIDE. Jamais fusionnee avec les metriques attention-only.

    python tools/cross_blocks.py --model meta-llama/Llama-3.2-1B-Instruct \\
        --tag llama3.2-1b --pairs 56 --st053 <dossier de la 0.5.3>
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

ARM_REF = "C3_opi"
ARM_ALT = ("C4", "C3_wiki")
ARMS = (ARM_REF,) + ARM_ALT

#: Les sept croisements necessaires. (signal, graphe).
COMBOS = ((ARM_REF, ARM_REF),
          ("C4", "C4"), (ARM_REF, "C4"), ("C4", ARM_REF),
          ("C3_wiki", "C3_wiki"), (ARM_REF, "C3_wiki"), ("C3_wiki", ARM_REF))

#: Regions entierement anterieures au slot dans l'ordre d'OPI.
AVANT_SLOT = ("task", "normal", "preamble")

TOL_ZERO = 1e-12


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--pairs", type=int, default=56)
    ap.add_argument("--per-stratum", type=int, default=2)
    ap.add_argument("--seed", type=int, default=2027)
    ap.add_argument("--max-tokens", type=int, default=512)
    ap.add_argument("--st053", default=None)
    ap.add_argument("--out", default=os.path.join("results", "opi",
                                                  "cross_blocks"))
    args = ap.parse_args()
    if args.st053:
        sys.path.insert(0, os.path.abspath(args.st053))
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

    import torch
    import spectral_trust
    from spectral_trust import GSPConfig, GraphConstructor
    from spectral_trust.dirichlet import dirichlet_blocks
    from spectral_trust.region_energy import block_names

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
    log(f"=== decomposition croisee par bloc  {args.model} -> {out_dir}")
    log(f"spectral-trust {spectral_trust.__version__}")
    gpath = os.path.join(os.path.dirname(spectral_trust.__file__), "graph.py")
    patched = "patch_graph_dtype.py" in open(gpath, encoding="utf-8").read()
    if not patched:
        raise SystemExit("appliquer spectral_trust_ext/patch_graph_dtype.py")
    log("correctif float64 : APPLIQUE")

    df, _ = ds.load()
    rows = sample(df, args.per_stratum, args.seed).head(args.pairs)
    template, _b = tm.resolve(args.model)
    cap = MultiCapturer(args.model, dtype="float32", max_tokens=args.max_tokens)
    L_layers, H_heads = cap.n_layers, cap.n_heads
    log(f"{L_layers} couches x {H_heads} tetes")

    gc_sym = GraphConstructor(GSPConfig(symmetrization="symmetric",
                                        normalization="sym",
                                        remove_self_loops=False))
    gc_non = GraphConstructor(GSPConfig(symmetrization="symmetric",
                                        normalization="none",
                                        remove_self_loops=False))

    def n_tokens(instruction, external):
        txt = cm.format_user_only(cap.tokenizer,
                                  cm.build_user_turn(instruction, external),
                                  template)
        return len(cap.tokenizer(txt, add_special_tokens=False)["input_ids"])

    fillers = {"C3_opi": ds.BenignFiller(df, seed=args.seed),
               "C3_wiki": wf.WikitextFiller(seed=args.seed)}

    BN = block_names(list(r6.REGIONS6))
    pre = [k for k, nm in enumerate(BN)
           if all(p in AVANT_SLOT for p in nm.split("--"))]
    log(f"{len(BN)} blocs ; {len(pre)} entierement anterieurs au slot : "
        f"{[BN[k] for k in pre]}")

    out, checks, pair_meta = [], [], []
    skipped = {"unmatched": 0, "too_long": 0, "audit": 0, "deltaN": 0,
               "labels": 0}

    for pi, (_, row) in enumerate(rows.iterrows()):
        pid = (f"{row.attack_type}__{row.task_type}__{row.injected_task}"
               f"__r{int(row.row_index)}")
        arms = build_arms(row, fillers, n_tokens)
        if arms is None:
            skipped["unmatched"] += 1
            continue
        if n_tokens(row.instruction, arms["texts"]["C4"]) > args.max_tokens:
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
        if not ok or not r6.audit(labels[ARM_REF], labels["C4"])["ok"]:
            skipped["audit"] += 1
            continue
        if len({len(segs[a]) for a in ARMS}) != 1:
            skipped["deltaN"] += 1
            continue
        # Les etiquettes doivent coincider position par position : sinon le
        # bloc (r, s) ne designe pas les memes tokens dans les trois bras et
        # le croisement n'a pas de sens.
        if not all(np.array_equal(labels[ARM_REF], labels[a]) for a in ARM_ALT):
            skipped["labels"] += 1
            log(f"[{pi + 1}] {pid} : etiquettes non alignees, ignoree")
            continue
        T = len(segs[ARM_REF])
        lab = labels[ARM_REF]
        itask = np.flatnonzero(lab == "task")

        A, X = {}, {}
        for arm in ARMS:
            att, hs = cap.forward(segs[arm], check=True, with_hidden=True)
            A[arm] = [np.asarray(a, dtype=np.float64) for a in att]
            X[arm] = [np.asarray(h, dtype=np.float64) for h in hs]
            del att, hs

        # -- controle 1 et 2 : causalite stricte sur les regions anterieures
        c12 = {"pair_id": pid, "T": T, "n_task": int(itask.size)}
        for a in ARM_ALT:
            dx = max(float(np.abs(X[ARM_REF][l][itask] - X[a][l][itask]).max())
                     for l in range(L_layers))
            da = max(float(np.abs(A[ARM_REF][l][:, itask][:, :, itask]
                                  - A[a][l][:, itask][:, :, itask]).max())
                     for l in range(L_layers))
            c12[f"max_dX_task_{a}"] = dx
            c12[f"max_dA_task_{a}"] = da

        # -- la decomposition, couche par couche
        zero_max = 0.0
        for l in range(L_layers):
            Et = {}
            for sg, gr in COMBOS:
                e, _d, _s, tot = dirichlet_blocks(
                    gc_sym, torch.as_tensor(A[gr][l]),
                    torch.as_tensor(X[sg][l]), list(lab), list(r6.REGIONS6))
                Et[(sg, gr)] = (e * T, tot)          # energie brute du bloc
            ref = Et[(ARM_REF, ARM_REF)][0]
            dref = Et[(ARM_REF, ARM_REF)][1]         # [H] total de reference
            den = np.where(np.abs(dref) > 1e-30, dref, np.nan)[:, None]

            # -- controle 3 : en 'none', les blocs anterieurs au slot ont
            #    un Delta EXACTEMENT nul. C'est la version juste du test.
            for a in ARM_ALT:
                en_r, _, _, _ = dirichlet_blocks(
                    gc_non, torch.as_tensor(A[ARM_REF][l]),
                    torch.as_tensor(X[ARM_REF][l]), list(lab),
                    list(r6.REGIONS6))
                en_a, _, _, _ = dirichlet_blocks(
                    gc_non, torch.as_tensor(A[a][l]),
                    torch.as_tensor(X[a][l]), list(lab), list(r6.REGIONS6))
                zero_max = max(zero_max, float(
                    np.abs(en_a[:, pre] - en_r[:, pre]).max()))

            for a in ARM_ALT:
                g = (Et[(ARM_REF, a)][0] - ref) / den        # graphe seul
                s = (Et[(a, ARM_REF)][0] - ref) / den        # signal seul
                t = (Et[(a, a)][0] - ref) / den              # total
                i = t - g - s
                for h in range(H_heads):
                    for k, nm in enumerate(BN):
                        out.append({"pair_id": pid, "contraste": a,
                                    "couche": l, "tete": h, "bloc": nm,
                                    "graphe": float(g[h, k]),
                                    "signal": float(s[h, k]),
                                    "interaction": float(i[h, k]),
                                    "total": float(t[h, k])})
        c12["max_zero_none_avant_slot"] = zero_max
        checks.append(c12)
        pair_meta.append({"pair_id": pid, "task_type": row.task_type,
                          "attack_type": row.attack_type,
                          "row_index": int(row.row_index), "n_tokens": T})
        log(f"[{pi + 1}/{len(rows)}] {pid}  T={T}  "
            f"dX_task<{max(c12[f'max_dX_task_{a}'] for a in ARM_ALT):.1e}  "
            f"zero_none<{zero_max:.1e}  {time.time() - t0:.0f}s")
        del A, X

    if not out:
        raise SystemExit("aucune paire retenue")
    pd.DataFrame(out).to_parquet(os.path.join(out_dir, "cross_blocks.parquet"),
                                 index=False)
    CK = pd.DataFrame(checks)
    CK.to_csv(os.path.join(out_dir, "controles.csv"), index=False)
    pd.DataFrame(pair_meta).to_csv(os.path.join(out_dir, "pairs.csv"),
                                   index=False)
    prov = {"spectral_trust": spectral_trust.__version__, "model": args.model,
            "tag": args.tag, "n_layers": L_layers, "n_heads": H_heads,
            "arm_ref": ARM_REF, "arms_alt": list(ARM_ALT),
            "combos": [list(c) for c in COMBOS], "blocks": BN,
            "blocs_avant_slot": [BN[k] for k in pre],
            "normalisation_terme": "E_tot(X_ref, L_ref)",
            "family": "hybrid", "signal": "hidden_states[l+1]",
            "normalization": "sym", "controle_zero": "none",
            "float64_patch": patched, "seed": args.seed,
            "per_stratum": args.per_stratum, "n_pairs_kept": len(pair_meta),
            "skipped": skipped, "python": platform.python_version(),
            "numpy": np.__version__, "torch": torch.__version__,
            "seconds": round(time.time() - t0, 1)}
    with open(os.path.join(out_dir, "provenance.json"), "w",
              encoding="utf-8") as fh:
        json.dump(prov, fh, indent=2, default=str)
    log(f"\n{len(pair_meta)} paires, {len(out)} lignes, "
        f"{time.time() - t0:.0f}s")
    log(f"controle causalite : max |dX_task| "
        f"{max(CK[[c for c in CK if c.startswith('max_dX')]].max()):.2e}, "
        f"max |dA_task| "
        f"{max(CK[[c for c in CK if c.startswith('max_dA')]].max()):.2e}")
    log(f"controle zero en 'none' avant slot : "
        f"{CK.max_zero_none_avant_slot.max():.2e} "
        f"(tolerance {TOL_ZERO:.0e})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
