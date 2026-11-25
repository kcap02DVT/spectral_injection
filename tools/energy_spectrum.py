"""Spectre d'energie du flux residuel sur le graphe d'attention, PAR TETE.

CE QUE C'EST. Pour chaque tete, la transformee de Fourier de graphe du
signal ``X = hidden_states[l + 1]`` sur la base propre de ``L_sym`` :

    m_k = || u_k^T X ||^2      k = 0 .. T-1

``m_k`` est l'energie du flux residuel portee par le mode de frequence
``lambda_k``. La base etant orthonormee, Parseval est une egalite exacte,
``sum_k m_k = ||X||_F^2``, et l'energie de Dirichlet que mesure la campagne
se reconstruit comme ``tr(X^T L X) = sum_k lambda_k m_k``. Le spectre
d'energie est donc la DECOMPOSITION EN FREQUENCE de cette mesure scalaire,
pas une mesure de plus.

FAMILLE HYBRIDE. Le signal est le flux residuel, pas l'attention : ces
quantites sortent du cadre attention-only et ne doivent jamais etre
fusionnees avec les metriques attention-seules dans un meme tableau ni une
meme correction BH.

LECTURE PRINCIPALE DECLAREE D'AVANCE. ``mass_frac = m_k / sum_j m_j``, et
non ``m_k``. Les masses brutes varient en ``c^2`` sous ``X -> cX``, donc un
Delta sur ``m_k`` peut etre porte par la seule norme du flux residuel ;
``mass_frac`` est invariante par changement d'echelle. Les deux sont
ecrites, la fraction est la mesure, la masse brute documente la norme.

PROFONDEUR PRINCIPALE DECLAREE D'AVANCE. ``round(0.25 * (L - 1))``, le
quart de la profondeur, pour coincider avec ``spectrum_at_depth.py``.
TOUTES les couches sont neanmoins calculees et ecrites : le choix est
declare avant de voir les resultats, et aucune couche n'est selectionnee
apres.

POURQUOI LA COMPARAISON PAR RANG EST DEFINIE. Le protocole impose
Delta N = 0, donc C4 et C3_opi ont exactement le meme nombre de tokens, les
deux laplaciens la meme taille, et les spectres le meme nombre de modes. Le
script VERIFIE cette egalite par paire et s'arrete sinon. Aucune
interpolation, aucun reechantillonnage.

AUCUNE MOYENNE ENTRE TETES. Une cellule = une (couche, tete). L'agregation
est laissee a l'analyse, qui la declare.

SOLVEUR DENSE OBLIGATOIRE. Avec ``eigen_solver="sparse"`` et T > 50, la
bibliotheque ne rend que ``num_eigenvalues // 2`` modes de chaque extremite
du spectre : Parseval devient faux, et toute normalisation en distribution
est biaisee. Le dense est force ici et le controle de Parseval le verifie.

    python tools/energy_spectrum.py --model meta-llama/Llama-3.2-1B-Instruct \\
        --tag llama3.2-1b --pairs 12 --st053 <dossier de la 0.5.3>
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

#: Bras compares. C4 = adverse, C3_opi = controle benin apparie.
ARM_ADV = "C4"
ARM_BEN = "C3_opi"
ARMS = (ARM_BEN, ARM_ADV)

#: Profondeur relative principale, declaree avant de voir les resultats.
PRIMARY_FRAC = 0.25

#: Tolerances declarees d'avance.
TOL_PARSEVAL = 1e-10          # |sum m_k / ||X||_F^2 - 1|
TOL_ORTHO = 1e-9              # ||U^T U - I||_max


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--pairs", type=int, default=12,
                    help="nombre de paires du plan a traiter")
    ap.add_argument("--per-stratum", type=int, default=2)
    ap.add_argument("--seed", type=int, default=2027)
    ap.add_argument("--max-tokens", type=int, default=512)
    ap.add_argument("--st053", default=None,
                    help="dossier contenant la spectral-trust a utiliser ; "
                         "place en tete de sys.path")
    ap.add_argument("--out", default=os.path.join("results", "opi",
                                                  "energy_spectrum"))
    args = ap.parse_args()

    if args.st053:
        sys.path.insert(0, os.path.abspath(args.st053))

    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

    import spectral_trust
    from spectral_trust import GSPConfig, GraphConstructor, SpectralAnalyzer

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
    log(f"=== spectre d'energie  {args.model} -> {out_dir}")
    log(f"spectral-trust {spectral_trust.__version__}  "
        f"({os.path.dirname(spectral_trust.__file__)})")

    # Le correctif float64 doit etre en place : sans lui construct_laplacian
    # degrade une entree float64 en float32 et Parseval ne tient plus qu'a
    # ~1e-8, donc TOL_PARSEVAL=1e-10 echouerait. Verifie, pas suppose.
    gpath = os.path.join(os.path.dirname(spectral_trust.__file__), "graph.py")
    with open(gpath, encoding="utf-8") as fh:
        patched = "patch_graph_dtype.py" in fh.read()
    log(f"correctif float64 de construct_laplacian : "
        f"{'APPLIQUE' if patched else 'ABSENT'}")
    if not patched:
        raise SystemExit("appliquer spectral_trust_ext/patch_graph_dtype.py "
                         "sur cette installation avant de mesurer")

    df, inj_templates = ds.load()
    rows = sample(df, args.per_stratum, args.seed).head(args.pairs)
    log(f"plan : {len(rows)} paires (seed {args.seed}, "
        f"per_stratum {args.per_stratum})")

    template, bos = tm.resolve(args.model)
    cap = MultiCapturer(args.model, dtype="float32", max_tokens=args.max_tokens)
    L_layers, H_heads = cap.n_layers, cap.n_heads
    primary_layer = int(round(PRIMARY_FRAC * (L_layers - 1)))
    log(f"modele : {L_layers} couches x {H_heads} tetes, device {cap.device}")
    log(f"couche principale declaree : {primary_layer}/{L_layers - 1} "
        f"({PRIMARY_FRAC:.2f} de la profondeur)")

    gc = GraphConstructor(GSPConfig(symmetrization="symmetric",
                                    normalization="sym",
                                    remove_self_loops=False))
    an = SpectralAnalyzer(GSPConfig(eigen_solver="dense"))

    def n_tokens(instruction, external):
        txt = cm.format_user_only(cap.tokenizer,
                                  cm.build_user_turn(instruction, external),
                                  template)
        return len(cap.tokenizer(txt, add_special_tokens=False)["input_ids"])

    fillers = {"C3_opi": ds.BenignFiller(df, seed=args.seed),
               "C3_wiki": wf.WikitextFiller(seed=args.seed)}

    import torch

    chunks, checks, pair_meta = [], [], []
    skipped = {"unmatched": 0, "too_long": 0, "audit": 0, "deltaN": 0}

    for pi, (_, row) in enumerate(rows.iterrows()):
        pid = (f"{row.attack_type}__{row.task_type}__{row.injected_task}"
               f"__r{int(row.row_index)}")
        arms = build_arms(row, fillers, n_tokens)
        if arms is None:
            skipped["unmatched"] += 1
            log(f"[{pi + 1}/{len(rows)}] {pid} : non apparie, ignore")
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
            segs[arm], labels[arm] = seg, lab
        if not ok:
            skipped["audit"] += 1
            log(f"[{pi + 1}/{len(rows)}] {pid} : segmentation incoherente")
            continue
        a = r6.audit(labels[ARM_BEN], labels[ARM_ADV])
        if not a["ok"]:
            skipped["audit"] += 1
            log(f"[{pi + 1}/{len(rows)}] {pid} : audit echoue, ignore")
            continue

        # Delta N = 0 : condition de la comparaison par rang. Verifiee.
        T_ben, T_adv = len(segs[ARM_BEN]), len(segs[ARM_ADV])
        if T_ben != T_adv:
            skipped["deltaN"] += 1
            log(f"[{pi + 1}/{len(rows)}] {pid} : Delta N = "
                f"{T_adv - T_ben} != 0, paire ignoree")
            continue
        T = T_adv

        worst_pars, worst_orth = 0.0, 0.0
        for arm in ARMS:
            att, hs = cap.forward(segs[arm], check=True, with_hidden=True)
            for l in range(len(att)):
                A = torch.as_tensor(np.asarray(att[l], dtype=np.float64))
                W = gc.symmetrize_attention(A.unsqueeze(0))[0]
                Lh = gc.construct_laplacian(W).cpu().numpy()
                X = np.asarray(hs[l], dtype=np.float64)
                fro = float((X ** 2).sum())
                for h in range(Lh.shape[0]):
                    vals, U = an.compute_eigendecomposition(Lh[h])
                    if vals.size != T:
                        raise SystemExit(
                            f"spectre tronque : {vals.size} modes pour T={T} "
                            f"-- eigen_solver n'est pas 'dense'")
                    m = SpectralAnalyzer.compute_spectral_masses(X, U)
                    # Controles, sur un echantillon de cellules pour le cout
                    if h == 0 or h == Lh.shape[0] - 1:
                        worst_pars = max(worst_pars,
                                         abs(m.sum() / fro - 1.0))
                        worst_orth = max(worst_orth, float(
                            np.abs(U.T @ U - np.eye(T)).max()))
                    chunks.append(pd.DataFrame({
                        "pair_id": pid, "arm": arm, "couche": l, "tete": h,
                        "rang": np.arange(T, dtype=np.int32),
                        "lambda": vals.astype(np.float64),
                        "mass": m.astype(np.float64),
                        "mass_frac": (m / m.sum()).astype(np.float64)}))
            del att, hs

        if worst_pars > TOL_PARSEVAL or worst_orth > TOL_ORTHO:
            raise SystemExit(
                f"controle d'integrite echoue sur {pid} : Parseval "
                f"{worst_pars:.2e} (tol {TOL_PARSEVAL:.0e}), orthonormalite "
                f"{worst_orth:.2e} (tol {TOL_ORTHO:.0e})")
        checks.append({"pair_id": pid, "parseval_max": worst_pars,
                       "ortho_max": worst_orth, "n_tokens": T})
        pair_meta.append({"pair_id": pid, "task_type": row.task_type,
                          "attack_type": row.attack_type,
                          "injected_task": row.injected_task,
                          "row_index": int(row.row_index), "n_tokens": T,
                          "n_slot": int((labels[ARM_ADV] == "slot").sum()),
                          "n_task": int((labels[ARM_ADV] == "task").sum())})
        log(f"[{pi + 1}/{len(rows)}] {pid}  T={T}  "
            f"Parseval<{worst_pars:.1e}  ortho<{worst_orth:.1e}  "
            f"{time.time() - t0:.0f}s")

    if not chunks:
        raise SystemExit("aucune paire retenue")

    M = pd.concat(chunks, ignore_index=True)
    M.to_parquet(os.path.join(out_dir, "masses_long.parquet"), index=False)
    pd.DataFrame(checks).to_csv(os.path.join(out_dir, "integrity.csv"),
                                index=False)
    pd.DataFrame(pair_meta).to_csv(os.path.join(out_dir, "pairs.csv"),
                                   index=False)
    prov = {"spectral_trust": spectral_trust.__version__,
            "spectral_trust_path": os.path.dirname(spectral_trust.__file__),
            "float64_patch": patched,
            "model": args.model, "tag": args.tag,
            "n_layers": L_layers, "n_heads": H_heads,
            "primary_frac": PRIMARY_FRAC, "primary_layer": primary_layer,
            "arms": list(ARMS), "signal": "hidden_states[l+1]",
            "family": "hybrid", "normalization": "sym",
            "eigen_solver": "dense",
            "tol_parseval": TOL_PARSEVAL, "tol_ortho": TOL_ORTHO,
            "seed": args.seed, "per_stratum": args.per_stratum,
            "n_pairs_kept": len(pair_meta), "skipped": skipped,
            "python": platform.python_version(),
            "numpy": np.__version__, "torch": torch.__version__,
            "seconds": round(time.time() - t0, 1)}
    with open(os.path.join(out_dir, "provenance.json"), "w",
              encoding="utf-8") as fh:
        json.dump(prov, fh, indent=2, default=str)

    log(f"\n{len(pair_meta)} paires, {len(M)} lignes, "
        f"{time.time() - t0:.0f}s")
    log(f"Parseval pire cas : {max(c['parseval_max'] for c in checks):.2e}")
    log(f"ecrits : {out_dir}/masses_long.parquet, integrity.csv, pairs.csv, "
        f"provenance.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
