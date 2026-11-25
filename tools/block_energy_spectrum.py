"""Spectre d'energie sur l'UNION de deux regions : les aretes croisees.

LA QUESTION. La vue par region seule supprime toutes les aretes
inter-regions, donc exactement la ou l'effet mesure dans ce depot se
trouve (``task--slot``). Ce script prend le sous-graphe induit sur
l'UNION de deux regions, qui conserve ``W[task, slot]`` et
``W[slot, task]``, et en rend le spectre.

    I = indices(R1) union indices(R2),  trie
    L_I = laplacien normalise du sous-graphe induit sur I
    m_k = || u_k^(I)T X[I] ||^2,   lambda_k^(I) dans [0, 2]

LES QUINZE BLOCS SONT CALCULES, PAS UN SEUL. Six regions donnent quinze
paires non ordonnees. Les quinze sont mesurees et ecrites ; ``task+slot``
est celle sur laquelle porte la question, les quatorze autres sont son
contexte. Aucune n'est choisie apres avoir vu les resultats.

TROIS BRAS. ``C3_opi`` reference, ``C4`` injecte, ``C3_wiki`` autre texte
benin de meme longueur en tokens. Le troisieme repond a la seule question
qui compte : un deplacement est-il du a l'injection, ou se produit-il des
qu'on change le texte du slot ?

CONTROLE DES ARETES CROISEES, a la couche declaree. Le meme spectre est
recalcule apres avoir mis ``W[R1, R2] = W[R2, R1] = 0``. L'ecart entre les
deux dit combien les aretes croisees pesent dans le spectre de l'union.
Si cet ecart est nul, l'union n'apprend rien de plus que les deux regions
separees et la question est close.

POURQUOI LE THEOREME DE CAUSALITE TOMBE ICI. Sur ``task`` seul, le
sous-graphe ne depend que de ``A[task, task]``, bit-identique entre les
bras : le spectre est identique, c'est force. Sur ``task union slot``, le
degre d'un token de ``task`` inclut ``W[task, slot]``, qui change avec le
slot. Le spectre PEUT donc bouger, et c'est tout l'interet.

FAMILLE HYBRIDE. Le signal est le flux residuel.

    python tools/block_energy_spectrum.py --model meta-llama/Llama-3.2-1B-Instruct
        --tag llama3.2-1b --pairs 16 --st053 <dossier 0.5.3>
"""

import argparse
import itertools
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
ARM_ADV = "C4"
ARM_BEN = "C3_wiki"
ARMS = (ARM_REF, ARM_ADV, ARM_BEN)

BLOC_DECLARE = "task+slot"
FRAC_COUCHE = 0.25
N_MIN_REGION = 2          # chaque region du bloc
N_MIN_UNION = 4           # et l'union
TOL_PARSEVAL = 1e-10


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--pairs", type=int, default=16)
    ap.add_argument("--per-stratum", type=int, default=2)
    ap.add_argument("--seed", type=int, default=2027)
    ap.add_argument("--max-tokens", type=int, default=512)
    ap.add_argument("--st053", default=None)
    ap.add_argument("--out", default=os.path.join("results", "opi",
                                                  "block_spectrum"))
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
    log(f"=== spectre sur l'UNION de deux regions  {args.model} -> {out_dir}")
    log(f"spectral-trust {spectral_trust.__version__}")
    gpath = os.path.join(os.path.dirname(spectral_trust.__file__), "graph.py")
    if "patch_graph_dtype.py" not in open(gpath, encoding="utf-8").read():
        raise SystemExit("appliquer spectral_trust_ext/patch_graph_dtype.py")
    log("correctif float64 : APPLIQUE")

    BLOCS = [f"{a}+{b}" for a, b in itertools.combinations(r6.REGIONS6, 2)]
    log(f"{len(BLOCS)} blocs : {BLOCS}")
    log(f"bloc declare : {BLOC_DECLARE} ; bras {ARMS}")

    df, _ = ds.load()
    rows = sample(df, args.per_stratum, args.seed).head(args.pairs)
    template, _b = tm.resolve(args.model)
    cap = MultiCapturer(args.model, dtype="float32",
                        max_tokens=args.max_tokens)
    L_layers, H_heads = cap.n_layers, cap.n_heads
    couche_p = int(round(FRAC_COUCHE * (L_layers - 1)))
    log(f"{L_layers} couches x {H_heads} tetes ; couche declaree {couche_p}")

    gc = GraphConstructor(GSPConfig(symmetrization="symmetric",
                                    normalization="sym",
                                    remove_self_loops=False))

    def n_tokens(instruction, external):
        txt = cm.format_user_only(cap.tokenizer,
                                  cm.build_user_turn(instruction, external),
                                  template)
        return len(cap.tokenizer(txt, add_special_tokens=False)["input_ids"])

    fillers = {"C3_opi": ds.BenignFiller(df, seed=args.seed),
               "C3_wiki": wf.WikitextFiller(seed=args.seed)}

    chunks, diag, meta = [], [], []
    skipped = {"unmatched": 0, "too_long": 0, "audit": 0, "deltaN": 0,
               "labels": 0}
    pire_pars = 0.0

    def spectre(Wh, I, X, croise_nul=None):
        """(lambda, mass_frac) du sous-graphe induit sur I."""
        sub = Wh[np.ix_(I, I)].copy()
        if croise_nul is not None:
            p, q = croise_nul
            sub[np.ix_(p, q)] = 0.0
            sub[np.ix_(q, p)] = 0.0
        Lr = gc.construct_laplacian(torch.as_tensor(sub))
        Lr = np.asarray(Lr[0] if np.ndim(Lr) == 3 else Lr, dtype=np.float64)
        vals, U = np.linalg.eigh((Lr + Lr.T) / 2.0)
        Xi = X[I]
        m = ((U.T @ Xi) ** 2).sum(axis=1)
        fro = float((Xi ** 2).sum())
        s = m.sum()
        return (np.clip(vals, 0.0, 2.0), m / s if s > 0 else m,
                abs(s / fro - 1.0) if fro > 0 else 0.0)

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
        idx = {r: np.flatnonzero(lab == r) for r in r6.REGIONS6}
        actifs = []
        for a, b in itertools.combinations(r6.REGIONS6, 2):
            if (idx[a].size >= N_MIN_REGION and idx[b].size >= N_MIN_REGION
                    and idx[a].size + idx[b].size >= N_MIN_UNION):
                actifs.append((a, b))
        tp = time.time()

        for arm in ARMS:
            att, hs = cap.forward(segs[arm], check=True, with_hidden=True)
            for l in range(L_layers):
                A = torch.as_tensor(np.asarray(att[l], dtype=np.float64))
                W = gc.symmetrize_attention(A)
                X = np.asarray(hs[l], dtype=np.float64)
                for h in range(H_heads):
                    Wh = np.asarray(W[h] if W.dim() == 3 else W,
                                    dtype=np.float64)
                    for a, b in actifs:
                        I = np.sort(np.concatenate([idx[a], idx[b]]))
                        vals, mf, pe = spectre(Wh, I, X)
                        pire_pars = max(pire_pars, pe)
                        chunks.append(pd.DataFrame({
                            "pair_id": pid, "arm": arm, "couche": l,
                            "tete": h, "bloc": f"{a}+{b}",
                            "n_union": int(I.size),
                            "rang": np.arange(I.size, dtype=np.int32),
                            "lambda": vals, "mass_frac": mf}))
                        if l != couche_p:
                            continue
                        # -- controle : les memes tokens, aretes croisees
                        #    annulees. L'ecart isole ce que les aretes
                        #    croisees apportent au spectre.
                        pos = {t: k for k, t in enumerate(I)}
                        pa = np.array([pos[t] for t in idx[a]])
                        pb = np.array([pos[t] for t in idx[b]])
                        v0, m0, _ = spectre(Wh, I, X, croise_nul=(pa, pb))
                        gr = np.union1d(vals, v0)
                        F1 = np.concatenate(([0.], np.cumsum(mf)))[
                            np.searchsorted(vals, gr, side="right")]
                        F0 = np.concatenate(([0.], np.cumsum(m0)))[
                            np.searchsorted(v0, gr, side="right")]
                        diag.append({
                            "pair_id": pid, "arm": arm, "couche": l,
                            "tete": h, "bloc": f"{a}+{b}",
                            "poids_croise": float(
                                Wh[np.ix_(idx[a], idx[b])].sum()
                                / max(Wh[np.ix_(I, I)].sum(), 1e-30)),
                            "D_sans_croise": float(np.abs(F1 - F0).max()),
                            "d_lambda_sans_croise": float(
                                np.abs(vals - v0).max())})
            del att, hs

        meta.append({"pair_id": pid, "T": len(segs[ARM_REF]),
                     "attack_type": row.attack_type,
                     "task_type": row.task_type,
                     "n_blocs": len(actifs),
                     **{f"n_{r}": int(idx[r].size) for r in r6.REGIONS6}})
        log(f"[{pi + 1}/{len(rows)}] {pid}  T={len(segs[ARM_REF])}  "
            f"{len(actifs)} blocs  {time.time() - tp:.1f}s")

    P = pd.concat(chunks, ignore_index=True)
    D = pd.DataFrame(diag)
    M = pd.DataFrame(meta)
    P.to_parquet(os.path.join(out_dir, "blocs_long.parquet"), index=False)
    D.to_csv(os.path.join(out_dir, "controle_croise.csv"), index=False)
    M.to_csv(os.path.join(out_dir, "paires.csv"), index=False)

    log("")
    log(f"{len(M)} paires retenues ; ignorees {skipped}")
    log(f"Parseval, pire ecart relatif : {pire_pars:.3e}")
    if pire_pars > TOL_PARSEVAL:
        raise SystemExit("controle de Parseval echoue")
    log("")
    log("-- ce que les aretes croisees apportent au spectre de l'union "
        f"(couche {couche_p}, bras {ARM_REF})")
    log(f"{'bloc':<20}{'poids croise':>14}{'D sans croise':>15}"
        f"{'max |d lambda|':>16}")
    dr = D[D.arm == ARM_REF]
    for bl, g in dr.groupby("bloc"):
        log(f"{bl:<20}{g.poids_croise.median():>14.4f}"
            f"{g.D_sans_croise.median():>15.4f}"
            f"{g.d_lambda_sans_croise.median():>16.4f}")

    json.dump({"model": args.model, "tag": args.tag,
               "spectral_trust": spectral_trust.__version__,
               "n_layers": L_layers, "n_heads": H_heads,
               "couche_declaree": couche_p, "bloc_declare": BLOC_DECLARE,
               "blocs": BLOCS, "arms": list(ARMS),
               "signal": "hidden_states[l+1]", "family": "hybrid",
               "normalization": "sym", "n_pairs_kept": len(M),
               "skipped": skipped, "parseval_max": pire_pars,
               "seed": args.seed, "per_stratum": args.per_stratum,
               "python": platform.python_version(),
               "secondes": round(time.time() - t0, 1)},
              open(os.path.join(out_dir, "provenance.json"), "w"), indent=2)
    log("")
    log(f"{len(P):,} lignes ; termine en {time.time() - t0:.0f}s")
    logf.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
