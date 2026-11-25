"""Essai des fonctionnalites de spectral-trust 0.5.3 sur les injections OPI.

CE QUE CE SCRIPT EST. Un essai des fonctions PUBLIEES en 0.5.3 qui n'ont
jamais ete utilisees dans ce depot, sur le corpus apparie OPI. Ce n'est pas
une campagne : la batterie est declaree ci-dessous AVANT de voir les
resultats, tout est rendu, rien n'est selectionne apres coup.

TROIS FAMILLES, JAMAIS FUSIONNEES.
  A  attention seule   -- ne depend que de A, donc du graphe
  B  hybride           -- depend des etats caches X (ecart assume au cadre
                          attention-only, cf. la note dirichlet-hybrid)
  C  ponderee valeurs  -- depend de A, de V et de W_O

CE QUI EST TESTE, et pourquoi c'est nouveau ici :
  A1 mag_lambda2        laplacien magnetique : la DIRECTION est preservee,
  A2 mag_lambda_max     contrairement a (A + A^T)/2. Repond a l'objection
  A3 mag_trou_spectral  de non-normalite (||AA*-A*A||/||A|| = 11,1).
  A4 op_pr              spectre de l'operateur d'attention non symetrise
  A5 op_pr_centre
  A6 op_contraction
  A7 forman_moy         courbure d'arete : une geometrie locale, pas un
  A8 forman_slot_med    spectre. Restreinte aux aretes du slot.
  A9 flow_desequilibre  communautes dirigees (k=2). MESURE, non presupposee.
  B1 mag_hfer           HFER directionnel
  B2 tv_dirigee         variation totale dirigee, normalisee par ||X||^2
  B3 bande_entropie     8 bandes, lambda_max DECLARE a 2.0
  B4 bande_part_haute   part des 4 bandes hautes
  B5 dirichlet_slot     energie ancree sur les lignes du slot (fast.py)
  C1 vw_part_bos        part de la masse ponderee-valeurs recue par BOS
  C2 vw_lambda2         lambda2 du laplacien sym du graphe ponderee-valeurs
  C3 vw_hfer            HFER sur ce graphe

PIEGE D'API, CONSTATE ET CONTOURNE. ``band_energies`` prend par defaut
``lambda_max`` = borne de Gershgorin de L, qui vaut 8,88 sur une cellule ou
le spectre est dans [0, 2] (un token quasi isole gonfle les entrees
normalisees). Les 4 bandes hautes lisent alors exactement 0 au lieu de
0,571. La borne est en outre DEPENDANTE DE L'ENTREE, donc les bandes ne
designent pas la meme chose entre deux prompts. ``lambda_max=2.0`` est
passe explicitement partout.

STATISTIQUE. Par tete (primaire, aucune moyenne entre tetes) : Wilcoxon
apparie bilateral sur les paires, dz de Cohen, Benjamini-Hochberg a q=0,05
DANS CHAQUE FAMILLE separement. Le contraste benin-contre-benin
(C3_wiki vs C3_opi) donne le rapport de specificite. Aucun seuil ajuste,
aucun classifieur, aucune selection de cellule apres resultats.

    python tools/test_st053_injection.py --model meta-llama/Llama-3.2-1B-Instruct
        --tag llama3.2-1b --pairs 16 --layers 4 --st053 <dossier 0.5.3>
"""

import argparse
import json
import os
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

Q_MAG = 0.25
CUTOFF = 0.1
N_BANDES = 8
LAMBDA_MAX = 2.0          # declare : le spectre de L_sym est dans [0, 2]
K_FLOT = 2
FDR_Q = 0.05

FAM = {
    "A_attention": ["mag_lambda2", "mag_lambda_max", "mag_trou_spectral",
                    "op_pr", "op_pr_centre", "op_contraction",
                    "forman_moy", "forman_slot_med", "flow_desequilibre"],
    "B_hybride": ["mag_hfer", "tv_dirigee", "bande_entropie",
                  "bande_part_haute", "dirichlet_slot"],
    "C_valeurs": ["vw_part_bos", "vw_lambda2", "vw_hfer"],
}
QUANTITES = [q for v in FAM.values() for q in v]


def mesures(DIR, OPS, FM, FA, gc, A_l, X_l, VW_l, islot, torch):
    """Les 17 quantites, pour chaque tete d'une couche. -> {nom: [H]}."""
    H = A_l.shape[0]
    out = {q: np.full(H, np.nan) for q in QUANTITES}
    nX2 = float(np.linalg.norm(np.asarray(X_l)) ** 2)
    Xt = torch.as_tensor(np.asarray(X_l))
    for h in range(H):
        Ah = torch.as_tensor(np.asarray(A_l[h]))
        # --- famille A : attention seule
        lam, _ = DIR.magnetic_spectrum(Ah, q=Q_MAG, normalized=True)
        lam = np.sort(np.asarray(lam, dtype=np.float64))
        out["mag_lambda2"][h] = lam[1]
        out["mag_lambda_max"][h] = lam[-1]
        out["mag_trou_spectral"][h] = lam[-1] - lam[1]
        om = OPS.operator_metrics(Ah)
        out["op_pr"][h] = float(
            np.asarray(om["participation_ratio"]).reshape(-1)[0])
        out["op_pr_centre"][h] = float(
            np.asarray(om["participation_ratio_centered"]).reshape(-1)[0])
        out["op_contraction"][h] = float(
            np.asarray(om["contraction"]).reshape(-1)[0])
        C = FM.forman_curvature(Ah, directed=True)
        out["forman_moy"][h] = float(np.asarray(
            FM.graph_curvature_summary(Ah, directed=True)[0]).reshape(-1)[0])
        qs = np.asarray(FM.forman_quantiles(C, tokens=islot.tolist())
                        ).reshape(-1)
        out["forman_slot_med"][h] = float(qs[len(qs) // 2])
        cl = DIR.hermitian_clustering(Ah, k=K_FLOT)
        out["flow_desequilibre"][h] = float(
            np.abs(np.asarray(cl.imbalance)).max())
        # --- famille B : hybride
        out["mag_hfer"][h] = float(np.asarray(
            DIR.magnetic_hfer(Xt, Ah, q=Q_MAG, normalized=True,
                              cutoff_ratio=CUTOFF)).reshape(-1)[0])
        P = DIR.directed_operator(Ah)
        out["tv_dirigee"][h] = float(np.asarray(
            DIR.directed_total_variation(Xt, P)).reshape(-1)[0]) / nX2
        Lh = gc.construct_laplacian(gc.symmetrize_attention(Ah))
        Lh = np.asarray(Lh[0] if np.ndim(Lh) == 3 else Lh, dtype=np.float64)
        r = FA.band_energies(Lh, np.asarray(X_l), edges=N_BANDES,
                             lambda_max=LAMBDA_MAX)
        E = np.asarray(r[0] if isinstance(r, tuple) else r,
                       dtype=np.float64).reshape(-1)
        out["bande_entropie"][h] = float(FA.band_entropy(E, normalize=True))
        out["bande_part_haute"][h] = float(E[N_BANDES // 2:].sum() / E.sum())
        Wh = (Ah + Ah.T) / 2
        out["dirichlet_slot"][h] = float(np.asarray(
            FA.row_dirichlet_energy(Wh, Xt, rows=islot)).reshape(-1)[0]) / nX2
        # --- famille C : ponderee par les valeurs
        Vh = torch.as_tensor(np.asarray(VW_l[h], dtype=np.float64))
        s = float(Vh.sum())
        out["vw_part_bos"][h] = float(Vh[:, 0].sum()) / s if s > 0 else np.nan
        Vn = Vh / s if s > 0 else Vh
        Lv = gc.construct_laplacian(gc.symmetrize_attention(Vn))
        Lv = np.asarray(Lv[0] if np.ndim(Lv) == 3 else Lv, dtype=np.float64)
        ev = np.sort(np.linalg.eigvalsh((Lv + Lv.T) / 2))
        out["vw_lambda2"][h] = float(ev[1])
        out["vw_hfer"][h] = float(np.asarray(
            DIR.magnetic_hfer(Xt, Vn, q=Q_MAG, normalized=True,
                              cutoff_ratio=CUTOFF)).reshape(-1)[0])
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--pairs", type=int, default=16)
    ap.add_argument("--per-stratum", type=int, default=2)
    ap.add_argument("--seed", type=int, default=2027)
    ap.add_argument("--max-tokens", type=int, default=512)
    ap.add_argument("--layers", default="4",
                    help="couches declarees, separees par des virgules")
    ap.add_argument("--st053", default=None)
    ap.add_argument("--out", default=os.path.join("results", "opi", "st053"))
    args = ap.parse_args()
    if args.st053:
        sys.path.insert(0, os.path.abspath(args.st053))
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

    import torch
    import spectral_trust as st
    from spectral_trust import GSPConfig, GraphConstructor
    from spectral_trust import directed as DIR, operator_spectra as OPS
    from spectral_trust import forman as FM, fast as FA, value_weighted as VW

    from spectral_injection.opi import capture as cm
    from spectral_injection.opi import dataset as ds
    from spectral_injection.opi import regions6 as r6
    from spectral_injection.opi import templates as tm
    from spectral_injection.opi import wikifiller as wf
    from run_region_energy import MultiCapturer, build_arms, sample

    couches = [int(x) for x in args.layers.split(",")]
    out_dir = os.path.join(args.out, args.tag)
    os.makedirs(out_dir, exist_ok=True)
    logf = open(os.path.join(out_dir, "run.log"), "w", encoding="utf-8",
                buffering=1)

    def log(*x):
        s = " ".join(str(v) for v in x)
        print(s, flush=True)
        logf.write(s + "\n")

    t0 = time.time()
    log(f"=== essai 0.5.3 sur injections  {args.model} -> {out_dir}")
    log(f"spectral-trust {st.__version__} <- {os.path.dirname(st.__file__)}")
    log(f"couches declarees {couches} ; q={Q_MAG} ; cutoff={CUTOFF} ; "
        f"lambda_max={LAMBDA_MAX} (declare, pas Gershgorin)")
    log(f"{len(QUANTITES)} quantites en 3 familles : "
        + " | ".join(f"{k}={len(v)}" for k, v in FAM.items()))

    df, _ = ds.load()
    rows = sample(df, args.per_stratum, args.seed).head(args.pairs)
    template, _b = tm.resolve(args.model)
    cap = MultiCapturer(args.model, dtype="float32",
                        max_tokens=args.max_tokens)
    log(f"{cap.n_layers} couches x {cap.n_heads} tetes")

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

    recs, meta = [], []
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
        islot = np.flatnonzero(labels[ARM_REF] == "slot")
        if islot.size == 0:
            skipped["labels"] += 1
            continue
        tp = time.time()
        for arm in ARMS:
            att, hs = cap.forward(segs[arm], check=True, with_hidden=True)
            vw = VW.value_weighted_attention(cap.model, segs[arm].input_ids,
                                             layers=couches, per_head=True,
                                             tokenizer=cap.tokenizer)
            for l in couches:
                V = vw[l]
                V = V[0] if V.dim() == 4 else V
                m = mesures(DIR, OPS, FM, FA, gc,
                            np.asarray(att[l], dtype=np.float64),
                            np.asarray(hs[l], dtype=np.float64),
                            V.to(torch.float64).numpy(), islot, torch)
                for h in range(cap.n_heads):
                    r = {"pair_id": pid, "arm": arm, "couche": l, "tete": h}
                    r.update({q: float(m[q][h]) for q in QUANTITES})
                    recs.append(r)
            del att, hs, vw
        meta.append({"pair_id": pid, "T": len(segs[ARM_REF]),
                     "n_slot": int(islot.size),
                     "attack_type": row.attack_type,
                     "task_type": row.task_type})
        log(f"[{pi + 1}/{len(rows)}] {pid}  T={len(segs[ARM_REF])} "
            f"slot={islot.size}  {time.time() - tp:.1f}s")

    D = pd.DataFrame(recs)
    M = pd.DataFrame(meta)
    D.to_parquet(os.path.join(out_dir, "mesures.parquet"), index=False)
    M.to_csv(os.path.join(out_dir, "paires.csv"), index=False)
    log(f"")
    log(f"{len(M)} paires retenues ; ignorees {skipped}")
    log(f"{len(D)} lignes (paire x bras x couche x tete)")
    nf = [f"{q}={int(np.sum(~np.isfinite(D[q])))}" for q in QUANTITES
          if int(np.sum(~np.isfinite(D[q]))) > 0]
    log("non finis : " + (", ".join(nf) if nf else "aucun"))
    json.dump({"model": args.model, "version": st.__version__,
               "pairs": len(M), "couches": couches, "q": Q_MAG,
               "cutoff": CUTOFF, "lambda_max": LAMBDA_MAX,
               "familles": FAM, "ignorees": skipped,
               "secondes": round(time.time() - t0, 1)},
              open(os.path.join(out_dir, "manifeste.json"), "w"), indent=2)
    log(f"")
    log(f"termine en {time.time() - t0:.0f}s")
    logf.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
