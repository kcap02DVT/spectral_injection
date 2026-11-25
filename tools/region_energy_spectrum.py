"""Spectre d'energie PAR REGION, au lieu du graphe entier.

DEUX VUES, qui ne repondent pas a la meme question.

VUE "propre" -- le graphe est restreint. Pour une region R, on prend le
SOUS-GRAPHE INDUIT sur les tokens de R (``GraphConstructor.subgraph_laplacian``
de la 0.5.3 : il extrait A[R, R] PUIS normalise, donc les degres sont ceux
du sous-graphe), on le diagonalise, et on projette X[R] sur SES PROPRES
vecteurs propres :

    m_k^R = || u_k^(R)T X[R] ||^2     avec  lambda_k^(R) dans [0, 2]

C'est la transposition exacte de la regle deja retenue : chaque signal sur
les vecteurs propres de son propre graphe, aucune valeur propre commune
imposee, meme echelle [0, 2] pour comparer a l'oeil.

CE QUE CETTE VUE JETTE, et il faut le dire. Le sous-graphe induit supprime
TOUTES les aretes inter-regions. Or l'effet mesure dans ce depot vit dans
les blocs croises -- ``task--slot``, graphe +3,19e-05, dz +1,73, 100 % des
paires. La vue "propre" ne peut donc pas le voir : par construction, elle
ne regarde que ce qui se passe A L'INTERIEUR d'une region.

VUE "partagee" -- le graphe reste entier. Memes vecteurs propres u_k du
graphe complet, mais le signal est restreint :

    m_k^(R,part) = || u_k^T X_R ||^2   avec X_R = X dont les lignes hors R
                                       sont mises a zero

Les six distributions vivent sur les MEMES frequences et somment chacune a
||X[R]||_F^2 (Parseval exact par region). Elle dit a quelle frequence du
graphe complet se loge l'energie de chaque region. Elle est calculee a la
seule couche declaree, pour le cout.

CONTROLE GRATUIT ET EXACT. L'attention est causale et ``task`` precede
``slot``, donc A[task, task] et X[task] sont bit-identiques entre les bras.
Le sous-graphe induit sur ``task`` ne depend que de A[task, task] : son
spectre et ses masses doivent etre IDENTIQUES entre C3_opi et C4, a zero
machine pres. C'est verifie et ecrit dans controles.csv. (Ce controle est
FAUX sur le graphe entier : la symetrisation couple les degres.)

FAMILLE HYBRIDE. Le signal est le flux residuel : jamais fusionne avec les
metriques attention-seules.

REGION PRINCIPALE DECLAREE : ``slot`` (deja declaree dans
signal_cleaning.py). COUCHE PRINCIPALE DECLAREE : round(0.25 * (L - 1)),
soit 4 sur un modele a 16 couches. Toutes les couches sont calculees pour
la vue "propre" ; aucune n'est choisie apres coup.

    python tools/region_energy_spectrum.py --model meta-llama/Llama-3.2-1B-Instruct
        --tag llama3.2-1b --pairs 16 --st053 <dossier 0.5.3>
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

ARM_BEN = "C3_opi"
ARM_ADV = "C4"
ARMS = (ARM_BEN, ARM_ADV)

REGION_PRINCIPALE = "slot"
FRAC_COUCHE = 0.25
N_MIN_REGION = 4          # en deca, le spectre n'a pas de sens
TOL_PARSEVAL = 1e-10
TOL_CONTROLE = 1e-12


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
                                                  "region_spectrum"))
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
    log(f"=== spectre d'energie PAR REGION  {args.model} -> {out_dir}")
    log(f"spectral-trust {spectral_trust.__version__}")
    gpath = os.path.join(os.path.dirname(spectral_trust.__file__), "graph.py")
    patched = "patch_graph_dtype.py" in open(gpath, encoding="utf-8").read()
    if not patched:
        raise SystemExit("appliquer spectral_trust_ext/patch_graph_dtype.py")
    log("correctif float64 : APPLIQUE")

    df, _ = ds.load()
    rows = sample(df, args.per_stratum, args.seed).head(args.pairs)
    template, _b = tm.resolve(args.model)
    cap = MultiCapturer(args.model, dtype="float32",
                        max_tokens=args.max_tokens)
    L_layers, H_heads = cap.n_layers, cap.n_heads
    couche_p = int(round(FRAC_COUCHE * (L_layers - 1)))
    log(f"{L_layers} couches x {H_heads} tetes ; couche declaree {couche_p} ; "
        f"region declaree '{REGION_PRINCIPALE}'")
    log(f"regions : {list(r6.REGIONS6)}")

    gc = GraphConstructor(GSPConfig(symmetrization="symmetric",
                                    normalization="sym",
                                    remove_self_loops=False))

    def n_tokens(instruction, external):
        txt = cm.format_user_only(cap.tokenizer,
                                  cm.build_user_turn(instruction, external),
                                  template)
        return len(cap.tokenizer(txt, add_special_tokens=False)["input_ids"])

    fillers = {"C3_opi": ds.BenignFiller(df, seed=args.seed)}

    prop, part, ctrl, meta = [], [], [], []
    skipped = {"unmatched": 0, "too_long": 0, "audit": 0, "deltaN": 0,
               "labels": 0}
    petites = {}
    pire_pars = 0.0

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
        if not ok or not r6.audit(labels[ARM_BEN], labels[ARM_ADV])["ok"]:
            skipped["audit"] += 1
            continue
        if len(segs[ARM_BEN]) != len(segs[ARM_ADV]):
            skipped["deltaN"] += 1
            continue
        if not np.array_equal(labels[ARM_BEN], labels[ARM_ADV]):
            skipped["labels"] += 1
            continue

        T = len(segs[ARM_BEN])
        lab = labels[ARM_BEN]
        idx = {r: np.flatnonzero(lab == r) for r in r6.REGIONS6}
        utiles = [r for r in r6.REGIONS6 if idx[r].size >= N_MIN_REGION]
        for r in r6.REGIONS6:
            if idx[r].size < N_MIN_REGION:
                petites[r] = petites.get(r, 0) + 1
        tp = time.time()

        garde = {}          # (arm, couche, tete, region) -> (lam, mfrac)
        for arm in ARMS:
            att, hs = cap.forward(segs[arm], check=True, with_hidden=True)
            for l in range(L_layers):
                A = torch.as_tensor(np.asarray(att[l], dtype=np.float64))
                W = gc.symmetrize_attention(A)
                X = np.asarray(hs[l], dtype=np.float64)
                for h in range(H_heads):
                    Wh = W[h] if W.dim() == 3 else W
                    # ---- vue "propre" : sous-graphe induit, base propre
                    for r in utiles:
                        I = idx[r]
                        Lr = gc.subgraph_laplacian(Wh, I.tolist())
                        Lr = np.asarray(Lr[0] if np.ndim(Lr) == 3 else Lr,
                                        dtype=np.float64)
                        Lr = (Lr + Lr.T) / 2.0
                        vals, U = np.linalg.eigh(Lr)
                        Xr = X[I]
                        m = ((U.T @ Xr) ** 2).sum(axis=1)
                        fro = float((Xr ** 2).sum())
                        if fro > 0:
                            pire_pars = max(pire_pars,
                                            abs(m.sum() / fro - 1.0))
                        s = m.sum()
                        mf = m / s if s > 0 else m
                        prop.append(pd.DataFrame({
                            "pair_id": pid, "arm": arm, "couche": l,
                            "tete": h, "region": r,
                            "n_region": int(I.size),
                            "rang": np.arange(I.size, dtype=np.int32),
                            "lambda": vals.astype(np.float64),
                            "mass_frac": mf.astype(np.float64)}))
                        if l == couche_p:
                            garde[(arm, l, h, r)] = (vals.copy(), m.copy())
                    # ---- vue "partagee" : graphe complet, signal restreint
                    if l == couche_p:
                        Lh = gc.construct_laplacian(Wh)
                        Lh = np.asarray(Lh[0] if np.ndim(Lh) == 3 else Lh,
                                        dtype=np.float64)
                        valsF, UF = np.linalg.eigh((Lh + Lh.T) / 2.0)
                        for r in utiles:
                            I = idx[r]
                            m = ((UF[I].T @ X[I]) ** 2).sum(axis=1)
                            s = m.sum()
                            part.append(pd.DataFrame({
                                "pair_id": pid, "arm": arm, "couche": l,
                                "tete": h, "region": r,
                                "n_region": int(I.size),
                                "rang": np.arange(T, dtype=np.int32),
                                "lambda": valsF.astype(np.float64),
                                "mass_frac": (m / s if s > 0 else m
                                              ).astype(np.float64)}))
            del att, hs

        # ---- controle exact : 'task' precede 'slot', donc identique
        for r in utiles:
            dl = dm = 0.0
            for h in range(H_heads):
                kb = (ARM_BEN, couche_p, h, r)
                ka = (ARM_ADV, couche_p, h, r)
                if kb in garde and ka in garde:
                    dl = max(dl, float(np.abs(garde[kb][0]
                                              - garde[ka][0]).max()))
                    dm = max(dm, float(np.abs(garde[kb][1]
                                              - garde[ka][1]).max()))
            ctrl.append({"pair_id": pid, "region": r, "n_region": int(idx[r].size),
                         "max_dlambda": dl, "max_dmasse": dm})
        meta.append({"pair_id": pid, "T": T, "attack_type": row.attack_type,
                     "task_type": row.task_type,
                     **{f"n_{r}": int(idx[r].size) for r in r6.REGIONS6}})
        log(f"[{pi + 1}/{len(rows)}] {pid}  T={T}  "
            + " ".join(f"{r}={idx[r].size}" for r in r6.REGIONS6)
            + f"  {time.time() - tp:.1f}s")

    P = pd.concat(prop, ignore_index=True)
    S = pd.concat(part, ignore_index=True)
    C = pd.DataFrame(ctrl)
    M = pd.DataFrame(meta)
    P.to_parquet(os.path.join(out_dir, "propre_long.parquet"), index=False)
    S.to_parquet(os.path.join(out_dir, "partagee_long.parquet"), index=False)
    C.to_csv(os.path.join(out_dir, "controles.csv"), index=False)
    M.to_csv(os.path.join(out_dir, "paires.csv"), index=False)

    log("")
    log(f"{len(M)} paires retenues ; ignorees {skipped}")
    log(f"regions trop petites (< {N_MIN_REGION} tokens), paires concernees : "
        f"{petites or 'aucune'}")
    log(f"Parseval, pire ecart relatif : {pire_pars:.3e} "
        f"(tolerance {TOL_PARSEVAL:.0e})")
    if pire_pars > TOL_PARSEVAL:
        raise SystemExit("controle de Parseval echoue")
    log("")
    log("-- controle de causalite, vue 'propre', couche declaree")
    log(f"{'region':<10}{'max |d lambda|':>16}{'max |d masse|':>16}"
        f"{'identique':>11}")
    for r, g in C.groupby("region", sort=False):
        dl, dm = float(g.max_dlambda.max()), float(g.max_dmasse.max())
        log(f"{r:<10}{dl:>16.3e}{dm:>16.3e}"
            f"{('OUI' if max(dl, dm) <= TOL_CONTROLE else 'non'):>11}")
    log("")
    log(f"vue propre   : {len(P):,} lignes")
    log(f"vue partagee : {len(S):,} lignes (couche {couche_p})")

    json.dump({"model": args.model, "tag": args.tag,
               "spectral_trust": spectral_trust.__version__,
               "n_layers": L_layers, "n_heads": H_heads,
               "couche_declaree": couche_p,
               "region_declaree": REGION_PRINCIPALE,
               "regions": list(r6.REGIONS6), "arms": list(ARMS),
               "signal": "hidden_states[l+1]", "family": "hybrid",
               "normalization": "sym", "n_min_region": N_MIN_REGION,
               "n_pairs_kept": len(M), "skipped": skipped,
               "parseval_max": pire_pars, "seed": args.seed,
               "per_stratum": args.per_stratum,
               "python": platform.python_version(),
               "secondes": round(time.time() - t0, 1)},
              open(os.path.join(out_dir, "provenance.json"), "w"), indent=2)
    log("")
    log(f"termine en {time.time() - t0:.0f}s")
    logf.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
