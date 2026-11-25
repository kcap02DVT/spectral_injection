"""Etapes 1, 3 et 4 du plan : nettoyer le signal, cibler, verifier.

CE QUE FAIT CE SCRIPT. Une seule passe de capture, puis, pour chaque
cellule (paire, bras, couche, tete) :

  etape 1  cinq variantes DECLAREES du signal, du brut au centre-sans-BOS
  etape 3  fraction haute frequence PAR REGION, et blocs de Dirichlet
  etape 4  nul par permutation, et decomposition croisee signal / graphe

FAMILLE HYBRIDE. Le signal est le flux residuel. Jamais fusionne avec les
metriques attention-only dans un tableau ni une correction BH.

CE QUI N'EST PAS FAIT, ET POURQUOI. Le plan demande a l'etape 1 de reperer
les autres puits d'attention et de << calculer avec et sans eux >>. Le
retrait des puits du GRAPHE est une operation que le protocole a
explicitement supprimee. Ce script ne touche donc jamais au graphe : le
masquage ne porte que sur le SIGNAL, X, le laplacien restant celui du
graphe complet. La variante V5 qui masque les puits detectes est calculee
comme DIAGNOSTIC, hors lecture principale et hors correction BH, et le
rapport la signale comme telle.

DECLARATIONS, toutes posees avant de voir le moindre resultat :

  couche principale    round(0.25 * (L - 1)), identique aux campagnes
  variante principale  V3 = BOS masque puis centrage sur les non-BOS
  regle conditionnelle V4 (cosinus) devient principale SI, sous V3, un
                       seul token porte encore plus de SEUIL_TOKEN de
                       ||X||_F^2 (mediane des cellules, couche principale)
  region principale    slot ; les six sont neanmoins calculees
  bloc principal       task--slot, celui deja declare dans l'extension
  coupure HF           lambda >= 1, le milieu de [0, 2]
  detection de puits   flux entrant moyen > SEUIL_PUITS / T
  permutation          N_PERM tirages, graine PERM_SEED, couche principale
  moitie de selection  les paires de rang pair une fois triees par
                       identifiant ; les impaires confirment

    python tools/signal_cleaning.py --model meta-llama/Llama-3.2-1B-Instruct \\
        --tag llama3.2-1b --pairs 16 --st053 <dossier de la 0.5.3>
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

ARM_ADV = "C4"
ARM_BEN = "C3_opi"
ARMS = (ARM_BEN, ARM_ADV)

PRIMARY_FRAC = 0.25
PRIMARY_VARIANT = "V3_sansBOS_centre"
PRIMARY_REGION = "slot"
PRIMARY_BLOCK = "task--slot"
HIGH_CUT = 1.0

#: Part de ||X||_F^2 qu'un seul token ne doit pas depasser apres nettoyage.
#: Milieu de la fourchette 20-30 % du plan. Declare, pas ajuste.
SEUIL_TOKEN = 0.25

#: Un token est un puits si son flux entrant moyen depasse ce multiple de
#: l'uniforme 1/T. Declare.
SEUIL_PUITS = 5.0

N_PERM = 200
PERM_SEED = 2027
TOL_PARSEVAL = 1e-10

VARIANTS = ("V0_brut", "V1_sansBOS", "V2_centre", "V3_sansBOS_centre",
            "V4_cosinus", "V5_sansPuits_DIAGNOSTIC")


def build_variants(X: np.ndarray, sinks: np.ndarray) -> dict:
    """Les six signaux declares, a partir du flux residuel brut [T, d].

    Le BOS est le token 0. << Masquer >> veut dire mettre sa ligne de X a
    zero : le graphe garde le token, son degre et ses aretes. Aucune
    operation ici ne touche au laplacien.
    """
    T = X.shape[0]
    keep = np.ones(T, dtype=bool)
    keep[0] = False                      # non-BOS

    V = {}
    V["V0_brut"] = X
    Z = X.copy()
    Z[0] = 0.0
    V["V1_sansBOS"] = Z
    V["V2_centre"] = X - X.mean(axis=0, keepdims=True)
    C = X - X[keep].mean(axis=0, keepdims=True)
    C[0] = 0.0
    V["V3_sansBOS_centre"] = C
    n = np.linalg.norm(C, axis=1, keepdims=True)
    V["V4_cosinus"] = np.divide(C, n, out=np.zeros_like(C), where=n > 1e-12)
    S = C.copy()
    S[sinks] = 0.0
    V["V5_sansPuits_DIAGNOSTIC"] = S
    return V


def detect_sinks(A_layer: np.ndarray):
    """Puits d'attention d'une couche : flux entrant moyen > SEUIL / T.

    ``A`` est [H, T, T] a lignes stochastiques, donc la moyenne sur les
    lignes donne une distribution sur les colonnes dont l'uniforme vaut
    1/T. Le BOS en fait partie et ressort toujours.

    Rend aussi le classement complet, pour que l'absence d'autres puits
    soit une CONSTATATION et non un effet du seuil : on voit ce que valent
    les candidats suivants.
    """
    inflow = np.asarray(A_layer, dtype=np.float64).mean(axis=(0, 1))
    return inflow > (SEUIL_PUITS / inflow.size), inflow


def cell_quantities(G: np.ndarray, Lh: np.ndarray, P: np.ndarray,
                    reg_idx: dict) -> dict:
    """Tout ce qui se lit sur la matrice de Gram, sans reprojeter X.

    ``G = S S^T`` [T, T] ne depend que de la couche et de la variante ;
    ``Lh`` et ``P`` ne dependent que de la tete. Les identites utilisees :

        ||S||_F^2          = tr(G)
        tr(S^T L S)        = <L, G>              energie de Dirichlet
        somme des m_k a    = <P, G>              P = U_H U_H^T, projecteur
          lambda >= 1                             sur les modes hauts
        ||(P S)_R||^2      = <P E_R P, G>        E_R selecteur de la region

    Le centroide du spectre d'energie vaut <L, G> / tr(G) : c'est le MEME
    nombre que le lissage. Les deux noms designent une seule quantite, et
    le script ne la compte qu'une fois.

    DEUX NORMALISATIONS DE LA MESURE PAR REGION, parce que l'enonce du plan
    n'en designe pas la meme dans sa phrase et dans sa formule :

        hf_R      = ||(P S)_R||^2 / ||S_R||^2    la formule. Ce n'est PAS
                    une fraction : P melange les lignes, donc l'energie
                    haute frequence posee sur R peut venir d'ailleurs et le
                    rapport depasse 1 sans difficulte.
        hfshare_R = ||(P S)_R||^2 / ||P S||^2    la phrase : << quelle
                    fraction de CETTE energie se trouve sur R >>. Bornee
                    par [0, 1] et de somme 1 sur les regions.

    Les deux sont rendues. La seconde est comparable entre paires, la
    premiere se lit comme une amplification.
    """
    fro = float(np.trace(G))
    if fro <= 0:
        return None
    E = float(np.sum(Lh * G))
    hi = float(np.sum(P * G))
    out = {"norme2": fro, "energie": E, "lissage": E / fro,
           "frac_high": hi / fro,
           "token_share_max": float(np.max(np.diag(G)) / fro)}
    for r, idx in reg_idx.items():
        if idx.size == 0:
            out[f"hf_{r}"] = np.nan
            out[f"hfshare_{r}"] = np.nan
            out[f"part_{r}"] = 0.0
            continue
        Pr = P[:, idx]                                   # [T, |R|]
        M = Pr @ Pr.T                                    # P E_R P
        num = float(np.sum(M * G))
        den = float(np.sum(np.diag(G)[idx]))
        out[f"hf_{r}"] = num / den if den > 0 else np.nan
        out[f"hfshare_{r}"] = num / hi if hi > 0 else np.nan
        out[f"part_{r}"] = den / fro
    return out


def permutation_null(G: np.ndarray, Lh: np.ndarray, P: np.ndarray,
                     idx: np.ndarray, rng) -> dict:
    """Nul par permutation des tokens du signal, a graphe FIXE.

    Melanger les lignes de ``S`` revient a permuter lignes et colonnes de
    ``G``. Toutes les quantites etant des produits scalaires <M, G>, un
    tirage coute un simple reindexage [T, T] : aucune reprojection, aucune
    rediagonalisation. C'est ce qui rend 200 tirages par cellule gratuits.

    Si la mesure ne bouge pas sous permutation, elle decrit le graphe et
    non le contenu porte par les tokens.
    """
    T = G.shape[0]
    d = np.diag(G)
    Pr = P[:, idx]
    M = Pr @ Pr.T
    hi0 = float(np.sum(P * G))
    obs_hf = float(np.sum(M * G)) / float(d[idx].sum())
    obs_hs = float(np.sum(M * G)) / hi0
    obs_fh = hi0 / float(d.sum())
    obs_li = float(np.sum(Lh * G)) / float(d.sum())
    hf, hs = np.empty(N_PERM), np.empty(N_PERM)
    fh, li = np.empty(N_PERM), np.empty(N_PERM)
    for b in range(N_PERM):
        p = rng.permutation(T)
        Gp = G[np.ix_(p, p)]
        dp = d[p]
        den = float(dp[idx].sum())
        num = float(np.sum(M * Gp))
        hip = float(np.sum(P * Gp))
        hf[b] = num / den if den > 0 else np.nan
        hs[b] = num / hip if hip > 0 else np.nan
        fh[b] = hip / float(dp.sum())
        li[b] = float(np.sum(Lh * Gp)) / float(dp.sum())
    o = {}
    for nm, obs, nul in (("hf", obs_hf, hf), ("hfshare", obs_hs, hs),
                         ("frac_high", obs_fh, fh), ("lissage", obs_li, li)):
        s = float(np.nanstd(nul))
        o[f"{nm}_obs"] = obs
        o[f"{nm}_nul_moy"] = float(np.nanmean(nul))
        o[f"{nm}_nul_ect"] = s
        o[f"{nm}_z"] = (obs - float(np.nanmean(nul))) / s if s > 1e-15 else np.nan
    return o


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
                                                  "signal_cleaning"))
    args = ap.parse_args()
    if args.st053:
        sys.path.insert(0, os.path.abspath(args.st053))
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

    import torch
    import spectral_trust
    from spectral_trust import GSPConfig, GraphConstructor, SpectralAnalyzer
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
    log(f"=== nettoyage du signal  {args.model} -> {out_dir}")
    log(f"spectral-trust {spectral_trust.__version__}")
    gpath = os.path.join(os.path.dirname(spectral_trust.__file__), "graph.py")
    patched = "patch_graph_dtype.py" in open(gpath, encoding="utf-8").read()
    log(f"correctif float64 : {'APPLIQUE' if patched else 'ABSENT'}")
    if not patched:
        raise SystemExit("appliquer spectral_trust_ext/patch_graph_dtype.py")

    df, _ = ds.load()
    rows = sample(df, args.per_stratum, args.seed).head(args.pairs)
    template, _bos = tm.resolve(args.model)
    cap = MultiCapturer(args.model, dtype="float32", max_tokens=args.max_tokens)
    L_layers, H_heads = cap.n_layers, cap.n_heads
    primary_layer = int(round(PRIMARY_FRAC * (L_layers - 1)))
    log(f"{L_layers} couches x {H_heads} tetes, couche principale "
        f"{primary_layer}")

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

    cells, perms, cross, blocks, sink_rows, pair_meta = [], [], [], [], [], []
    skipped = {"unmatched": 0, "too_long": 0, "audit": 0, "deltaN": 0}
    rng = np.random.default_rng(PERM_SEED)

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
            segs[arm], labels[arm] = seg, lab
        if not ok or not r6.audit(labels[ARM_BEN], labels[ARM_ADV])["ok"]:
            skipped["audit"] += 1
            continue
        if len(segs[ARM_BEN]) != len(segs[ARM_ADV]):
            skipped["deltaN"] += 1
            continue
        T = len(segs[ARM_ADV])

        # -- une passe par bras, tout garde en memoire pour l'etape 4b
        store = {}
        for arm in ARMS:
            att, hs = cap.forward(segs[arm], check=True, with_hidden=True)
            lab = np.asarray(labels[arm])
            reg_idx = {r: np.flatnonzero(lab == r) for r in r6.REGIONS6}
            per_layer = {}
            for l in range(len(att)):
                A = np.asarray(att[l], dtype=np.float64)
                sinks, inflow = detect_sinks(A)
                At = torch.as_tensor(A)
                W = gc.symmetrize_attention(At.unsqueeze(0))[0]
                Lh = gc.construct_laplacian(W).cpu().numpy()      # [H, T, T]
                X = np.asarray(hs[l], dtype=np.float64)
                V = build_variants(X, sinks)
                Gs = {k: s @ s.T for k, s in V.items()}
                eig = []
                for h in range(H_heads):
                    vals, U = an.compute_eigendecomposition(Lh[h])
                    if vals.size != T:
                        raise SystemExit("spectre tronque : solveur non dense")
                    Uh = U[:, vals >= HIGH_CUT]
                    eig.append((vals, Uh @ Uh.T))
                    for vn, G in Gs.items():
                        q = cell_quantities(G, Lh[h], eig[h][1], reg_idx)
                        if q is None:
                            continue
                        cells.append({"pair_id": pid, "arm": arm, "couche": l,
                                      "tete": h, "variante": vn,
                                      "n_puits": int(sinks.sum()), **q})
                if l == primary_layer:
                    per_layer["L"] = Lh
                    per_layer["G"] = Gs[PRIMARY_VARIANT]
                    per_layer["eig"] = eig
                    per_layer["reg"] = reg_idx
                    # -- etape 4a, nul par permutation
                    ri = reg_idx[PRIMARY_REGION]
                    if ri.size:
                        for h in range(H_heads):
                            o = permutation_null(Gs[PRIMARY_VARIANT], Lh[h],
                                                 eig[h][1], ri, rng)
                            perms.append({"pair_id": pid, "arm": arm,
                                          "tete": h, **o})
                    # -- etape 3, blocs de Dirichlet sur le signal nettoye
                    e, dens, sh, tot = dirichlet_blocks(
                        gc, At, torch.as_tensor(V[PRIMARY_VARIANT]),
                        list(lab), list(r6.REGIONS6))
                    bn = block_names(list(r6.REGIONS6))
                    for h in range(H_heads):
                        for k, nm in enumerate(bn):
                            blocks.append({"pair_id": pid, "arm": arm,
                                           "tete": h, "bloc": nm,
                                           "share": float(sh[h, k]),
                                           "density": float(dens[h, k])})
                    # Les 6 plus forts flux entrants, qu'ils franchissent
                    # le seuil ou non : l'absence d'autres puits doit se
                    # constater, pas se deduire du seuil.
                    top = np.argsort(inflow)[::-1][:6]
                    toks = segs[arm].tokens
                    for rk, j in enumerate(top):
                        sink_rows.append({
                            "pair_id": pid, "arm": arm, "couche": l, "T": T,
                            "rang": rk, "token_idx": int(j),
                            "token": str(toks[j]).encode(
                                "ascii", "backslashreplace").decode(),
                            "region": str(lab[j]),
                            "flux_entrant": float(inflow[j]),
                            "x_uniforme": float(inflow[j] * T),
                            "est_puits": bool(sinks[j]),
                            "n_puits_couche": int(sinks.sum())})
            store[arm] = per_layer
            del att, hs

        # -- etape 4b, decomposition croisee a la couche principale
        Gb, Ga = store[ARM_BEN]["G"], store[ARM_ADV]["G"]
        Lb, La = store[ARM_BEN]["L"], store[ARM_ADV]["L"]
        nb, na = float(np.trace(Gb)), float(np.trace(Ga))
        for h in range(H_heads):
            ebb = float(np.sum(Lb[h] * Gb))
            eaa = float(np.sum(La[h] * Ga))
            eba = float(np.sum(La[h] * Gb))      # signal benin, graphe adverse
            eab = float(np.sum(Lb[h] * Ga))      # signal adverse, graphe benin
            cross.append({"pair_id": pid, "tete": h,
                          "E_ben_ben": ebb, "E_adv_adv": eaa,
                          "E_ben_sur_graphe_adv": eba,
                          "E_adv_sur_graphe_ben": eab,
                          "S_ben_ben": ebb / nb, "S_adv_adv": eaa / na,
                          "S_ben_sur_graphe_adv": eba / nb,
                          "S_adv_sur_graphe_ben": eab / na})
        pair_meta.append({"pair_id": pid, "task_type": row.task_type,
                          "attack_type": row.attack_type,
                          "row_index": int(row.row_index), "n_tokens": T,
                          "n_slot": int((np.asarray(labels[ARM_ADV])
                                         == "slot").sum())})
        log(f"[{pi + 1}/{len(rows)}] {pid}  T={T}  {time.time() - t0:.0f}s")

    if not cells:
        raise SystemExit("aucune paire retenue")

    pd.DataFrame(cells).to_parquet(
        os.path.join(out_dir, "cells.parquet"), index=False)
    pd.DataFrame(perms).to_parquet(
        os.path.join(out_dir, "permutation.parquet"), index=False)
    pd.DataFrame(cross).to_parquet(
        os.path.join(out_dir, "cross.parquet"), index=False)
    pd.DataFrame(blocks).to_parquet(
        os.path.join(out_dir, "blocks.parquet"), index=False)
    pd.DataFrame(sink_rows).to_csv(
        os.path.join(out_dir, "sinks.csv"), index=False)
    pd.DataFrame(pair_meta).to_csv(
        os.path.join(out_dir, "pairs.csv"), index=False)
    prov = {"spectral_trust": spectral_trust.__version__,
            "float64_patch": patched, "model": args.model, "tag": args.tag,
            "n_layers": L_layers, "n_heads": H_heads,
            "primary_layer": primary_layer, "primary_frac": PRIMARY_FRAC,
            "primary_variant": PRIMARY_VARIANT,
            "primary_region": PRIMARY_REGION, "primary_block": PRIMARY_BLOCK,
            "variants": list(VARIANTS), "high_cut": HIGH_CUT,
            "seuil_token": SEUIL_TOKEN, "seuil_puits": SEUIL_PUITS,
            "n_perm": N_PERM, "perm_seed": PERM_SEED,
            "family": "hybrid", "signal": "hidden_states[l+1]",
            "normalization": "sym", "eigen_solver": "dense",
            "graphe_intact": True,
            "seed": args.seed, "per_stratum": args.per_stratum,
            "n_pairs_kept": len(pair_meta), "skipped": skipped,
            "python": platform.python_version(), "numpy": np.__version__,
            "torch": torch.__version__, "seconds": round(time.time() - t0, 1)}
    with open(os.path.join(out_dir, "provenance.json"), "w",
              encoding="utf-8") as fh:
        json.dump(prov, fh, indent=2, default=str)
    log(f"\n{len(pair_meta)} paires, {len(cells)} lignes de cellules, "
        f"{time.time() - t0:.0f}s")
    log(f"ecrits dans {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
