"""Campagne multi-modeles : energie regionale sous injection, a Delta N = 0.

CE QUI EST TESTE. Pour chaque ligne d'OPI on construit trois prompts de
LONGUEUR EN TOKENS IDENTIQUE :

    C4         normal + P_atk + I_inj          + D_inj      (adverse)
    C3_opi     normal + P_atk + filler_opi     + D_inj      (controle)
    C3_wiki    normal + P_atk + filler_wiki    + D_inj      (controle)

Seul le contenu de la region ``slot`` change. Les cinq autres regions sont
identiques caractere par caractere, et l'audit de segmentation verifie que
leurs tailles en tokens coincident. Toute difference observee est donc une
REDISTRIBUTION, pas un effet de taille de graphe.

DEUX REMPLISSAGES, PAS UN. ``C3_opi`` tire le remplissage dans OPI (le plus
proche en distribution, mais 35 % des remplissages commencent par un
marqueur structure du type << Sentence1: ' >>). ``C3_wiki`` le tire de
wikitext (prose continue, mais ecart de domaine). Les deux bras sont
analyses separement : une signature presente sous un seul des deux serait
un artefact du remplissage, pas une propriete de l'injection.

TOUTE L'ALGEBRE DE GRAPHE PASSE PAR ``spectral_trust``. Le module
``spectral_trust.region_energy`` appelle ``GraphConstructor`` pour la
symetrisation et la normalisation ; l'equivalence avec le calcul artisanal
du pilote precedent est etablie a 9e-16 par
``tools/test_region_energy_equiv.py``.

L, H ET LE TEMPLATE SONT DETECTES. Le nombre de couches et de tetes est lu
dans la configuration du modele charge (y compris les configurations
imbriquees du type ``Mistral3Config.text_config``), jamais suppose. Le
template conversationnel vient du registre declare -- aucun bloc systeme
n'est ajoute.

SORTIES, par modele, dans ``results/opi/region_energy/<modele>/`` :

    blocks_long.parquet     une ligne par (paire, bras, couche, tete, bloc)
    flows_long.parquet      une ligne par (paire, bras, couche, tete, flux)
    scalars_long.parquet    offdiag / loop / variance par cellule
    pairs.csv               journal d'appariement, comptes par region, ASR
    segmentation_audit.csv  tous les controles de frontiere
    provenance.json         versions, seed, template, L, H, residus
"""

import argparse
import json
import os
import platform
import sys
import time
from typing import Dict, List

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import spectral_trust
from spectral_trust import GSPConfig, GraphConstructor
from spectral_trust.region_energy import (block_names, check_variance_identity,
                                          region_energy_all_layers)
from spectral_trust.dirichlet import (PRIMARY_BLOCK, check_block_decomposition,
                                      dirichlet_blocks_all_layers,
                                      dirichlet_energy_all_layers,
                                      head_aggregation_gap)

from spectral_injection.opi import asr as asr_mod
from spectral_injection.opi import capture as cm
from spectral_injection.opi import dataset as ds
from spectral_injection.opi import regions6 as r6
from spectral_injection.opi import templates as tm
from spectral_injection.opi import wikifiller as wf

ARMS = ("C3_opi", "C3_wiki", "C4")
CONTROL_ARMS = ("C3_opi", "C3_wiki")


# --------------------------------------------------------------------------
# Capture tolerante aux configurations imbriquees
# --------------------------------------------------------------------------
class MultiCapturer(cm.Capturer):
    """``Capturer`` + lecture de L/H dans les configurations imbriquees.

    ``capture.Capturer`` lit ``cfg.num_hidden_layers``, ce qui echoue sur
    les configurations enveloppes (``Mistral3Config`` expose le modele de
    texte sous ``text_config``). La classe parente est appelee telle
    quelle ; en cas d'``AttributeError`` -- levee APRES le chargement du
    modele, donc avec ``self.model`` deja en place -- l'initialisation est
    terminee ici. ``capture.py`` reste bit-identique au gel du pilote.
    """

    def __init__(self, model_id: str, **kw):
        try:
            super().__init__(model_id, **kw)
        except AttributeError:
            if not hasattr(self, "model"):
                raise
            cfg = self.model.config
            tc = getattr(cfg, "text_config", None) or cfg
            self.n_layers = int(tc.num_hidden_layers)
            self.n_heads = int(tc.num_attention_heads)
            self.n_kv_heads = int(getattr(tc, "num_key_value_heads",
                                          self.n_heads))
            self.nested_config = type(cfg).__name__


# --------------------------------------------------------------------------
# Echantillonnage
# --------------------------------------------------------------------------
def sample(df: pd.DataFrame, per_stratum: int, seed: int) -> pd.DataFrame:
    """Plan stratifie ``task_type x attack_type`` = 7 x 4 = 28 cellules.

    Reutilise ``dataset.sample_rows``, dont la graine est stable entre
    processus (BLAKE2b, pas ``hash()``).
    """
    return ds.sample_rows(df, per_stratum=per_stratum, seed=seed)


# --------------------------------------------------------------------------
# Une paire
# --------------------------------------------------------------------------
def build_arms(row, fillers, n_tokens) -> Dict:
    """Les trois prompts et leurs composants, ou None si non apparie."""
    base = n_tokens(row.instruction, ds.external_C2(row))
    target = n_tokens(row.instruction, ds.external_C4(row)) - base
    out = {"target_tokens": int(target), "fillers": {}}
    texts = {"C4": ds.external_C4(row)}
    comps = {"C4": r6.components(row, row.I_inj)}
    for arm, filler in fillers.items():
        txt, diag = filler.fit(row, n_tokens, target)
        out["fillers"][arm] = diag
        if txt is None:
            return None
        texts[arm] = ds.external_C3(row, txt)
        comps[arm] = r6.components(row, txt)
        if n_tokens(row.instruction, texts[arm]) != n_tokens(row.instruction,
                                                             texts["C4"]):
            return None
    out["texts"] = texts
    out["comps"] = comps
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--tag", required=True, help="nom de dossier court")
    ap.add_argument("--per-stratum", type=int, default=2,
                    help="2 -> 56 paires (28 strates)")
    ap.add_argument("--seed", type=int, default=2027)
    ap.add_argument("--max-tokens", type=int, default=512)
    ap.add_argument("--dirichlet", action="store_true",
                    help="energie de Dirichlet par tete sur le flux residuel. "
                         "METRIQUE HYBRIDE : exige les etats caches, donc "
                         "sort du cadre attention-only")
    ap.add_argument("--asr", action="store_true",
                    help="genere aussi la sortie et classe le detournement")
    ap.add_argument("--save-raw", type=int, default=0,
                    help="nombre de paires dont l'attention brute est gardee")
    ap.add_argument("--limit", type=int, default=0,
                    help="ne traiter que les N premieres paires du plan "
                         "(test de bon fonctionnement uniquement)")
    ap.add_argument("--out", default=os.path.join("results", "opi",
                                                  "region_energy"))
    args = ap.parse_args()

    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

    out_dir = os.path.join(args.out, args.tag)
    os.makedirs(out_dir, exist_ok=True)
    log_path = os.path.join(out_dir, "run.log")
    logf = open(log_path, "w", encoding="utf-8", buffering=1)

    def log(*x):
        s = " ".join(str(v) for v in x)
        print(s, flush=True)
        logf.write(s + "\n")

    t0 = time.time()
    log(f"=== {args.model}  ->  {out_dir}")

    # -- controles prealables, avant tout calcul
    ls = asr_mod.check_label_sets()
    if not ls["disjoint"]:
        raise SystemExit(f"ensembles d'etiquettes non disjoints : "
                         f"{ls['violations']}")

    df, inj_templates = ds.load()
    rows = sample(df, args.per_stratum, args.seed)
    if args.limit:
        rows = rows.head(args.limit)
        log(f"ATTENTION : --limit {args.limit}, plan tronque (test seulement)")
    log(f"plan : {len(rows)} paires, {args.per_stratum} par strate, "
        f"{rows.task_type.nunique()} taches x {rows.attack_type.nunique()} "
        f"attaques")

    template, bos = tm.resolve(args.model)
    cap = MultiCapturer(args.model, dtype="float32",
                        max_tokens=args.max_tokens)
    tverify = tm.verify(cap.tokenizer, template, bos)
    log(f"modele : {cap.n_layers} couches x {cap.n_heads} tetes "
        f"(kv={cap.n_kv_heads}), device {cap.device}")
    log(f"template : {tverify['name']}, "
        f"{tverify['n_template_tokens']} tokens de format")

    gc = GraphConstructor(GSPConfig(symmetrization="symmetric",
                                    normalization="sym",
                                    remove_self_loops=False))
    # Laplacien combinatoire, uniquement pour l'energie de Dirichlet. Il ne
    # sert pas a mesurer : ``L = D - W`` est lineaire en W, donc la moyenne
    # des energies par tete y egale l'energie du graphe moyen. C'est
    # precisement pour cela qu'il est calcule -- controle d'integrite du
    # chemin par tete, comme ``h2_dirichlet_comb`` controle ``h2_cut``.
    gc_comb = GraphConstructor(GSPConfig(symmetrization="symmetric",
                                         normalization="none",
                                         remove_self_loops=False))
    gen = asr_mod.Generator(cap.model, cap.tokenizer) if args.asr else None

    def n_tokens(instruction, external):
        txt = cm.format_user_only(cap.tokenizer,
                                  cm.build_user_turn(instruction, external),
                                  template)
        return len(cap.tokenizer(txt, add_special_tokens=False)["input_ids"])

    fillers = {"C3_opi": ds.BenignFiller(df, seed=args.seed),
               "C3_wiki": wf.WikitextFiller(seed=args.seed)}

    BN = block_names(r6.REGIONS6)
    blocks, flows, scalars, pair_rows, audits = [], [], [], [], []
    dirichlet, dir_blocks, linearity_checks, decomp_checks = [], [], [], []
    identity_residuals = []
    skipped = {"unmatched": 0, "too_long": 0, "audit": 0}

    # ``sample_id`` n'identifie PAS une ligne : avec per_stratum >= 2, deux
    # lignes de la meme strate (task_type, attack_type) peuvent partager un
    # sample_id en differant par injected_task. Un identifiant bati sur
    # attack__task__sample_id collisionnait alors, et le pivot de l'analyse
    # ECRASAIT silencieusement une paire par l'autre -- defaut constate sur
    # qwen2.5-0.5b, 55 paires ecrites, 53 identifiants distincts.
    # ``row_index`` est l'index unique de la ligne dans OPI (0..33599).
    seen = set()
    for pi, (_, row) in enumerate(rows.iterrows()):
        pid = (f"{row.attack_type}__{row.task_type}__{row.injected_task}"
               f"__r{int(row.row_index)}")
        if pid in seen:
            raise SystemExit(f"identifiant de paire duplique : {pid}")
        seen.add(pid)
        arms = build_arms(row, fillers, n_tokens)
        if arms is None:
            skipped["unmatched"] += 1
            log(f"[{pi + 1}/{len(rows)}] {pid} : non apparie, ignore")
            continue
        if n_tokens(row.instruction, arms["texts"]["C4"]) > args.max_tokens:
            skipped["too_long"] += 1
            log(f"[{pi + 1}/{len(rows)}] {pid} : trop long, ignore")
            continue

        segs, labels, gens = {}, {}, {}
        ok = True
        for arm in ARMS:
            seg = cm.segment(cap.tokenizer, row.instruction,
                             arms["texts"][arm], template)
            lab = r6.label_tokens(seg, row.instruction, arms["comps"][arm])
            three = r6.to_three(lab)
            if not np.array_equal(three, seg.regions):
                ok = False
                log(f"    {arm} : la segmentation a 6 regions n'est pas un "
                    f"raffinement de celle a 3")
                break
            segs[arm], labels[arm] = seg, lab

        if not ok:
            skipped["audit"] += 1
            continue

        aud = {"pair_id": pid}
        for ctl in CONTROL_ARMS:
            a = r6.audit(labels[ctl], labels["C4"])
            aud[f"ok_{ctl}"] = a["ok"]
            aud[f"detail_{ctl}"] = json.dumps(a, default=str)
        audits.append(aud)
        if not all(aud[f"ok_{c}"] for c in CONTROL_ARMS):
            skipped["audit"] += 1
            log(f"[{pi + 1}/{len(rows)}] {pid} : audit de segmentation "
                f"echoue, ignore")
            continue

        for arm in ARMS:
            seg = segs[arm]
            if args.dirichlet:
                att, hs = cap.forward(seg, check=True, with_hidden=True)
            else:
                att, hs = cap.forward(seg, check=True), None
            if not identity_residuals:
                identity_residuals.append(
                    check_variance_identity(gc, att[len(att) // 2]))
            re_ = region_energy_all_layers(gc, att, labels[arm], r6.REGIONS6)

            L, H = re_.n_layers, re_.n_heads
            li = np.repeat(np.arange(L), H)
            hi = np.tile(np.arange(H), L)
            for k, bn in enumerate(BN):
                blocks.append(pd.DataFrame({
                    "pair_id": pid, "arm": arm, "couche": li, "tete": hi,
                    "bloc": bn,
                    "energy": re_.energy[:, :, k].ravel(),
                    "density": re_.density[:, :, k].ravel(),
                    "share": re_.share[:, :, k].ravel()}))
            for i, q in enumerate(r6.REGIONS6):
                for j, kk in enumerate(r6.REGIONS6):
                    flows.append(pd.DataFrame({
                        "pair_id": pid, "arm": arm, "couche": li, "tete": hi,
                        "query": q, "key": kk,
                        "flow": re_.flow[:, :, i, j].ravel()}))
            scalars.append(pd.DataFrame({
                "pair_id": pid, "arm": arm, "couche": li, "tete": hi,
                "offdiag": re_.offdiag.ravel(), "loop": re_.loop.ravel(),
                "variance": re_.variance.ravel()}))

            if hs is not None:
                # Une cellule = une tete, jamais un graphe moyenne. La
                # moyenne sur les tetes est laissee a l'analyse, qui la
                # declare ; ce fichier garde la carte complete.
                de_sym = dirichlet_energy_all_layers(gc, att, hs)
                de_comb = dirichlet_energy_all_layers(gc_comb, att, hs)
                dirichlet.append(pd.DataFrame({
                    "pair_id": pid, "arm": arm, "couche": li, "tete": hi,
                    "energy_sym": de_sym.energy.ravel(),
                    "smoothness_sym": de_sym.smoothness.ravel(),
                    "energy_comb": de_comb.energy.ravel(),
                    "smoothness_comb": de_comb.smoothness.ravel(),
                    "signal_norm": np.repeat(de_sym.signal_norm, H)}))
                # Controle d'integrite, une couche mediane par paire-bras :
                # l'ecart 'none' doit etre nul, l'ecart 'sym' ne doit pas
                # l'etre. Le second n'est pas une erreur, c'est la mesure.
                mid = len(att) // 2
                for nm, c in (("none", gc_comb), ("sym", gc)):
                    g = head_aggregation_gap(c, att[mid], hs[mid])
                    linearity_checks.append(
                        {"pair_id": pid, "arm": arm, "couche": mid,
                         "normalization": nm, "rel": g["rel"],
                         "abs": g["abs"], "head_std": g["head_std"]})

                # Decomposition de l'energie de Dirichlet sur les 21 blocs
                # regionaux. La ``share`` est invariante par changement
                # d'echelle du flux residuel, donc elle ne peut pas etre
                # portee par la norme -- c'est la lecture principale.
                db = dirichlet_blocks_all_layers(gc, att, hs, labels[arm],
                                                 r6.REGIONS6)
                for k, bn in enumerate(db.block_names):
                    dir_blocks.append(pd.DataFrame({
                        "pair_id": pid, "arm": arm, "couche": li, "tete": hi,
                        "bloc": bn,
                        "d_energy": db.energy[:, :, k].ravel(),
                        "d_density": db.density[:, :, k].ravel(),
                        "d_share": db.share[:, :, k].ravel()}))
                if not decomp_checks:
                    for nm, c in (("sym", gc), ("none", gc_comb)):
                        decomp_checks.append(dict(
                            check_block_decomposition(c, att[mid], hs[mid],
                                                      labels[arm],
                                                      r6.REGIONS6),
                            primary_block=PRIMARY_BLOCK))

            if gen is not None:
                gens[arm] = gen.generate(seg.input_ids)

            if pi < args.save_raw:
                d = os.path.join(out_dir, "raw", pid)
                os.makedirs(d, exist_ok=True)
                np.savez_compressed(
                    os.path.join(d, arm + ".npz"),
                    **{f"layer_{l:02d}": a.astype(np.float32)
                       for l, a in enumerate(att)},
                    tokens=np.array(seg.tokens, str), regions=labels[arm])
            del att, hs

        rec = {"pair_id": pid, "task_type": row.task_type,
               "attack_type": row.attack_type,
               "injected_task": row.injected_task,
               "row_index": int(row.row_index),
               "n_tokens": len(segs["C4"]),
               "target_tokens_slot": arms["target_tokens"]}
        for arm in ARMS:
            for r, v in r6.counts(labels[arm]).items():
                rec[f"{arm}_n_{r}"] = v
        for arm in CONTROL_ARMS:
            rec[f"{arm}_candidates"] = arms["fillers"][arm]["candidates_tried"]
        if gen is not None:
            for arm in ARMS:
                cl = asr_mod.classify(gens[arm], row.task_type,
                                      row.injected_task)
                rec[f"gen_{arm}"] = gens[arm].replace("\n", " ")[:160]
                rec[f"asr_defined_{arm}"] = cl["defined"]
                rec[f"asr_hijacked_{arm}"] = cl["hijacked"]
                rec[f"asr_label_{arm}"] = cl["matched_label"]
        pair_rows.append(rec)

        log(f"[{pi + 1}/{len(rows)}] {pid}  T={len(segs['C4'])}  "
            f"slot={rec['C4_n_slot']}  "
            f"{'hijack=' + str(rec.get('asr_hijacked_C4')) if gen else ''}  "
            f"{time.time() - t0:.0f}s")

    if not pair_rows:
        raise SystemExit("aucune paire retenue")

    log(f"\nconcatenation : {len(blocks)} fragments de blocs")
    pd.concat(blocks, ignore_index=True).to_parquet(
        os.path.join(out_dir, "blocks_long.parquet"), index=False)
    pd.concat(flows, ignore_index=True).to_parquet(
        os.path.join(out_dir, "flows_long.parquet"), index=False)
    pd.concat(scalars, ignore_index=True).to_parquet(
        os.path.join(out_dir, "scalars_long.parquet"), index=False)
    if dirichlet:
        pd.concat(dirichlet, ignore_index=True).to_parquet(
            os.path.join(out_dir, "dirichlet_long.parquet"), index=False)
        pd.concat(dir_blocks, ignore_index=True).to_parquet(
            os.path.join(out_dir, "dirichlet_blocks_long.parquet"), index=False)
        for c in decomp_checks:
            log(f"Dirichlet, somme des 21 blocs vs tr(X^T L X), "
                f"{c['normalization']} : {c['rel_max']:.2e}")
        LC = pd.DataFrame(linearity_checks)
        LC.to_csv(os.path.join(out_dir, "dirichlet_linearity.csv"),
                  index=False)
        for nm, g in LC.groupby("normalization"):
            log(f"Dirichlet, ecart moyenne-par-tete vs graphe-moyen, "
                f"{nm} : max {g['rel'].max():.3e}  median "
                f"{g['rel'].median():.3e}")
    pairs_df = pd.DataFrame(pair_rows)
    pairs_df.to_csv(os.path.join(out_dir, "pairs.csv"), index=False,
                    encoding="utf-8")
    pd.DataFrame(audits).to_csv(
        os.path.join(out_dir, "segmentation_audit.csv"), index=False,
        encoding="utf-8")

    prov = {
        "model": args.model, "tag": args.tag,
        "n_layers": cap.n_layers, "n_heads": cap.n_heads,
        "n_kv_heads": cap.n_kv_heads, "device": cap.device,
        "nested_config": getattr(cap, "nested_config", None),
        "template": tverify, "bos_marker": bos,
        "regions": list(r6.REGIONS6), "blocks": BN, "arms": list(ARMS),
        "n_pairs_planned": int(len(rows)), "n_pairs_kept": len(pair_rows),
        "skipped": skipped,
        "seed": args.seed, "per_stratum": args.per_stratum,
        "max_tokens": args.max_tokens,
        "asr": bool(args.asr),
        "dirichlet": bool(args.dirichlet),
        "dirichlet_note": (
            "signal = flux residuel hidden_states[l+1], donc famille HYBRIDE "
            "et non attention-only ; energie par tete, jamais sur un graphe "
            "moyenne entre tetes ; 'comb' (L = D - W) est un controle "
            "d'integrite, pas une mesure, car la moyenne par tete y est "
            "identique a l'energie du graphe moyen par linearite"
        ) if args.dirichlet else None,
        "dirichlet_primary_block": PRIMARY_BLOCK if args.dirichlet else None,
        "dirichlet_decomposition_check": decomp_checks or None,
        "dirichlet_linearity": ({
            nm: {"rel_max": float(g["rel"].max()),
                 "rel_median": float(g["rel"].median())}
            for nm, g in pd.DataFrame(linearity_checks).groupby(
                "normalization")} if linearity_checks else None),
        "asr_criterion": {"label_sets": asr_mod.LABEL_SETS,
                          "open_tasks": list(asr_mod.OPEN_TASKS),
                          "max_new_tokens": asr_mod.MAX_NEW_TOKENS,
                          "disjoint": ls["disjoint"]},
        "wikitext_pool": len(fillers["C3_wiki"].pool),
        "identity_residual": identity_residuals[0] if identity_residuals else None,
        "graph": {"symmetrization": "symmetric", "normalization": "sym",
                  "remove_self_loops": False,
                  "note": "M = I - construct_laplacian(symmetrize(A)) ; "
                          "Q = M^2, Q_ii = 0"},
        "spectral_trust_version": spectral_trust.__version__,
        "spectral_trust_path": os.path.dirname(spectral_trust.__file__),
        "numpy": np.__version__, "pandas": pd.__version__,
        "python": platform.python_version(),
        "elapsed_s": round(time.time() - t0, 1),
    }
    if args.asr:
        prov["asr_summary"] = {
            arm: {"n_defined": int(pairs_df[f"asr_defined_{arm}"].sum()),
                  "n_hijacked": int(pairs_df[f"asr_hijacked_{arm}"]
                                    .fillna(False).astype(bool).sum())}
            for arm in ARMS}
    with open(os.path.join(out_dir, "provenance.json"), "w",
              encoding="utf-8") as fh:
        json.dump(prov, fh, indent=2, ensure_ascii=False, default=str)

    log(f"\nTERMINE {out_dir}  {len(pair_rows)} paires retenues, "
        f"ignorees {skipped}, residu d'identite "
        f"{prov['identity_residual']:.2e}, {prov['elapsed_s']:.0f}s")
    if args.asr:
        log(f"detournement : {prov['asr_summary']}")
    logf.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
