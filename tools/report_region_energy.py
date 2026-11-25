"""Rapport comparatif multi-modeles, genere depuis les CSV d'analyse.

N'INTERPRETE RIEN QU'IL NE MESURE. Le rapport rend, pour chaque quantite :
le Delta moyen, son intervalle de confiance bootstrap stratifie, la taille
d'effet appariee, le q de Benjamini-Hochberg, l'accord de signe entre
cellules, et -- surtout -- si la quantite est SIGNIFICATIVE ET DE MEME
SIGNE sous les deux remplissages de controle, dans combien de modeles.

Le critere de retenue est declare avant lecture des resultats :

    ROBUSTE dans un modele   significative (BH q<0,05) dans les deux bras
                             de controle, et de meme signe dans les deux
    REPRODUITE               robuste dans TOUS les modeles analyses, avec
                             le meme signe partout

Une quantite significative dans un seul bras est rendue mais signalee comme
possible artefact de remplissage. Aucune quantite n'est promue apres coup.
"""

import argparse
import json
import os
import sys
from typing import Dict, List

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

FAMILIES = {
    "block_energy": "Energie des blocs (somme de Q sur le bloc / N)",
    "block_share": "Part du bloc dans l'energie hors diagonale",
    "directed_flow": "Flux d'attention dirige (A brute, par token requete)",
    "scalar": "Scalaires globaux",
}


def g(x, fmt="{:+.4g}"):
    return fmt.format(float(x)) if np.isfinite(x) else "n.d."


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=os.path.join("results", "opi",
                                                   "region_energy"))
    ap.add_argument("--top", type=int, default=10)
    args = ap.parse_args()
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

    tests = pd.read_csv(os.path.join(args.root, "all_tests.csv"))
    conc = pd.read_csv(os.path.join(args.root, "all_concordance.csv"))
    rep = pd.read_csv(os.path.join(args.root, "reproducibility.csv"))

    provs: List[Dict] = []
    for t in sorted(os.listdir(args.root)):
        p = os.path.join(args.root, t, "provenance.json")
        if os.path.exists(p):
            provs.append(json.load(open(p, encoding="utf-8")))
    models = [p["tag"] for p in provs]

    L = []
    L.append("# Energie regionale sous injection : campagne multi-modeles\n")
    L.append("## Protocole\n")
    L.append("Trois prompts par exemple, de **longueur en tokens "
             "identique**, ne differant que par le contenu de la region "
             "`slot` :\n")
    L.append("```")
    L.append("C4       normal + P_atk + I_inj       + D_inj   adverse")
    L.append("C3_opi   normal + P_atk + filler_opi  + D_inj   controle, "
             "remplissage tire d'OPI")
    L.append("C3_wiki  normal + P_atk + filler_wiki + D_inj   controle, "
             "remplissage tire de wikitext")
    L.append("```\n")
    L.append("Les six regions sont `task`, `normal`, `preamble`, `slot`, "
             "`tail`, `template`. L'audit verifie, pour chaque exemple, que "
             "les **six** tailles en tokens coincident entre les trois "
             "bras : toute difference mesuree est donc une redistribution, "
             "pas un effet de taille de graphe.\n")
    L.append("Toute l'algebre de graphe passe par `spectral_trust` "
             "(`GraphConstructor.symmetrize_attention`, "
             "`construct_laplacian`), via le module "
             "`spectral_trust.region_energy`. L'equivalence avec le calcul "
             "artisanal du pilote precedent est etablie a 9,2e-16.\n")

    L.append("## Modeles\n")
    L.append("| modele | identifiant | couches x tetes | cellules | paires | "
             "template | residu d'identite |")
    L.append("|---|---|---:|---:|---:|---|---:|")
    for p in provs:
        L.append(f"| {p['tag']} | `{p['model']}` | "
                 f"{p['n_layers']} x {p['n_heads']} | "
                 f"{p['n_layers'] * p['n_heads']} | "
                 f"{p['n_pairs_kept']}/{p['n_pairs_planned']} | "
                 f"`{p['template']['name']}` | "
                 f"{float(p['identity_residual']):.1e} |")
    L.append("")

    if any(p.get("asr") for p in provs):
        L.append("## Succes de l'injection (detournement de tache)\n")
        L.append("Critere declare avant execution : la sortie contient une "
                 "etiquette de la tache **injectee** et non de la tache "
                 "legitime, appariement le plus long d'abord. Non defini "
                 "quand la tache injectee est a sortie libre (`gigaword`, "
                 "`jfleg`), alors exclu et jamais compte comme echec.\n")
        L.append("| modele | C4 detourne / defini | C3_opi | C3_wiki |")
        L.append("|---|---:|---:|---:|")
        for p in provs:
            s = p.get("asr_summary") or {}
            def cell(arm):
                a = s.get(arm) or {}
                nd, nh = a.get("n_defined"), a.get("n_hijacked")
                if not nd:
                    return "n.d."
                return f"{nh}/{nd} ({nh / nd:.0%})"
            L.append(f"| {p['tag']} | **{cell('C4')}** | {cell('C3_opi')} | "
                     f"{cell('C3_wiki')} |")
        L.append("")
        L.append("Les deux colonnes de controle sont la lecture utile : un "
                 "taux non nul en C3 signalerait que le critere capte un "
                 "biais de vocabulaire du modele plutot que l'injection.\n")

    # ----------------------------------------------------------------
    # Couplage au succes de l'injection. Place AVANT la liste des
    # quantites reproduites, et non en annexe : sans lui, cette liste se
    # lirait comme un ensemble de signatures du detournement, ce qu'elle
    # n'est pas. La presence d'une instruction injectee et son succes sont
    # deux choses differentes, et presque toutes ces quantites ne
    # repondent qu'a la premiere.
    # ----------------------------------------------------------------
    wpath = os.path.join(args.root, "asr_within_model.csv")
    bpath = os.path.join(args.root, "asr_between_models.csv")
    if os.path.exists(wpath):
        import json as _json
        from scipy.stats import binomtest

        W = pd.read_csv(wpath)
        B = (pd.read_csv(bpath) if os.path.exists(bpath)
             else pd.DataFrame(columns=["quantity", "spearman_rho", "p"]))
        L.append("## Presence de l'injection, ou succes de l'injection ?\n")
        L.append("Distinction decisive. Une quantite peut reagir au seul "
                 "fait qu'un texte imperatif occupe le `slot`, sans rien "
                 "dire de la question qui compte : le modele y obeit-il ? "
                 "Le test separe, A L'INTERIEUR de chaque modele, les "
                 "paires dont C4 a detourne la tache de celles ou "
                 "l'attaque a echoue. Les deux groupes partagent le "
                 "modele, le template et le plan de sondage.\n")
        rows = []
        for q, grp in W.groupby("quantity"):
            diffs = []
            for _, r in grp.iterrows():
                diffs += [d for _, d, _ in _json.loads(r.strata_detail)]
            pos = sum(1 for d in diffs if d > 0)
            sp = (binomtest(pos, len(diffs), 0.5).pvalue if diffs
                  else float("nan"))
            bb = B[B.quantity == q]
            rows.append({
                "quantity": q,
                "n_sig": int((grp.mannwhitney_p < 0.05).sum()),
                "n_models": int(len(grp)),
                "strata_pos": pos, "strata_n": len(diffs),
                "sign_p": sp,
                "rho": float(bb.spearman_rho.iloc[0]) if len(bb) else float("nan"),
            })
        T = pd.DataFrame(rows).sort_values(
            ["n_sig", "strata_pos"], ascending=False)
        L.append("| quantite | significatif | signes intra-strate | "
                 "test des signes | rho inter-modeles |")
        L.append("|---|---:|---:|---:|---:|")
        for _, r in T.iterrows():
            bold = "**" if r.n_sig == r.n_models and r.n_models > 1 else ""
            L.append(f"| {bold}`{r.quantity}`{bold} | "
                     f"{bold}{r.n_sig}/{r.n_models}{bold} | "
                     f"{r.strata_pos}/{r.strata_n} | "
                     f"{r.sign_p:.3f} | {r.rho:+.3f} |")
        L.append("")
        best = T[T.n_sig == T.n_models]
        if len(best) and T.n_models.max() > 1:
            L.append("A LIRE AINSI. Les quantites dont la colonne "
                     "<< significatif >> n'est pas pleine reagissent a la "
                     "PRESENCE de l'instruction injectee et non a son "
                     "succes : leur Delta est le meme que l'attaque "
                     "aboutisse ou non. Elles ne sont donc pas des "
                     "signatures du detournement.\n")
            for _, r in best.iterrows():
                L.append(f"- `{r.quantity}` est l'exception : "
                         f"significative dans {r.n_sig}/{r.n_models} "
                         f"modeles, {r.strata_pos}/{r.strata_n} "
                         f"comparaisons intra-strate de meme signe "
                         f"(test des signes p = {r.sign_p:.3f}).")
            L.append("")
        L.append("CONFOND NON LEVE. Le succes n'est pas assigne au hasard : "
                 "il depend fortement du type d'attaque. La composition des "
                 "groupes est dans `asr_within_model.csv`. Sur au moins un "
                 "modele le detournement est presque entierement determine "
                 "par `attack_type` (toutes les attaques `combine` "
                 "reussissent, aucune `escape`), si bien qu'AUCUNE strate "
                 "n'y est testable. Ailleurs, une a deux strates seulement "
                 "le sont, et aucun test intra-strate n'atteint 0,05. Ce "
                 "qui subsiste est la constance du SIGNE a type d'attaque "
                 "constant, pas une significativite intra-strate. Trancher "
                 "exigerait un plan equilibre sur `attack_type x issue`, "
                 "c'est-a-dire un nouvel echantillonnage, pas une "
                 "re-analyse.\n")
        L.append("PORTEE DE `template -> slot`. La region `template` "
                 "contient les marqueurs de format, dont la POSITION DE "
                 "GENERATION. L'attention emise depuis la est ce que le "
                 "modele lit au moment de decider quoi produire. La "
                 "quantite est donc en partie un releve COMPORTEMENTAL : "
                 "si le modele est sur le point d'emettre la reponse de la "
                 "tache injectee, il est presque necessaire que cette "
                 "position lise l'instruction injectee. Elle localise ou la "
                 "decision se lit, et se mesure en une seule passe avant "
                 "sans generer, mais elle n'est pas une propriete "
                 "structurelle du graphe au meme titre qu'une statistique "
                 "spectrale.\n")

    L.append("## Reproductibilite inter-modeles\n")
    L.append(f"Quantites robustes dans **tous** les {len(models)} modeles "
             f"(significatives BH q<0,05 ET de meme signe sous les deux "
             f"remplissages, dans chaque modele) :\n")
    una = rep[rep.unanimous]
    if una.empty:
        L.append("**Aucune.** Aucune quantite ne passe le critere dans "
                 "tous les modeles.\n")
    else:
        L.append("| quantite | famille | Delta median (opi) | "
                 "Delta median (wiki) | signe | accord de cellules |")
        L.append("|---|---|---:|---:|:--:|---:|")
        for _, r in una.iterrows():
            sg = "+" if r.n_positive > 0 else "-"
            L.append(f"| `{r.quantity}` | {r.family} | "
                     f"{g(r.median_delta_opi)} | {g(r.median_delta_wiki)} | "
                     f"**{sg}** | {r.median_agreement:.0%} |")
        L.append("")

    L.append("### Quantites robustes dans une partie des modeles seulement\n")
    part = rep[(~rep.unanimous) & (rep.n_robust > 0)].head(args.top)
    if part.empty:
        L.append("Aucune.\n")
    else:
        L.append("| quantite | famille | robuste dans | + | - | "
                 "Delta median (opi) |")
        L.append("|---|---|---:|---:|---:|---:|")
        for _, r in part.iterrows():
            L.append(f"| `{r.quantity}` | {r.family} | "
                     f"{r.n_robust}/{r.n_models} | {r.n_positive} | "
                     f"{r.n_negative} | {g(r.median_delta_opi)} |")
        L.append("")

    L.append("### Significatives sous UN SEUL remplissage\n")
    L.append("A lire comme de possibles artefacts du texte de controle, "
             "pas comme une signature de l'injection.\n")
    one = conc[(conc.sig_opi ^ conc.sig_wiki)]
    if one.empty:
        L.append("Aucune : toute quantite significative l'est sous les "
                 "deux remplissages.\n")
    else:
        cnt = (one.groupby(["family", "quantity"])
                 .size().sort_values(ascending=False).head(args.top))
        L.append("| quantite | famille | modeles concernes |")
        L.append("|---|---|---:|")
        for (fam, q), n in cnt.items():
            L.append(f"| `{q}` | {fam} | {n} |")
        L.append("")

    for fam, title in FAMILIES.items():
        sub = tests[(tests.family == fam)]
        if sub.empty:
            continue
        L.append(f"## {title}\n")
        L.append("Delta = C4 - C3, moyenne sur les paires ; IC 95 % par "
                 "bootstrap stratifie par type d'attaque, 10 000 rejets. "
                 "Bras `C3_opi`.\n")
        s = sub[sub.arm == "C3_opi"]
        key = (s.groupby("quantity").delta_mean
                .apply(lambda x: np.abs(x).median())
                .sort_values(ascending=False).head(args.top).index.tolist())
        L.append("| quantite | " + " | ".join(models) + " |")
        L.append("|---|" + "|".join(["---:"] * len(models)) + "|")
        for q in key:
            cells = []
            for m in models:
                r = s[(s.quantity == q) & (s.model == m)]
                if r.empty:
                    cells.append("n.d.")
                    continue
                r = r.iloc[0]
                star = "**" if r.sig_bh else ""
                cells.append(f"{star}{g(r.delta_mean)}{star}")
            L.append(f"| `{q}` | " + " | ".join(cells) + " |")
        L.append("")
        L.append("Gras = BH q<0,05 dans ce modele.\n")

    L.append("## Limites\n")
    L.append("- `slot` designe l'instruction injectee en C4 et le "
             "remplissage en C3. Un Delta sur un bloc impliquant `slot` "
             "compare donc deux textes differents de meme longueur, pas la "
             "presence contre l'absence de texte.")
    L.append("- Pour `ignore` et `combine`, `preamble` contient deja une "
             "directive adversariale, conservee a l'identique dans les "
             "trois bras. Le contraste y isole la seule instruction de "
             "tache injectee ; les moyennes par strate sont dans "
             "`all_tests.csv`.")
    L.append("- L'unite d'inference est la paire. Les L x H cellules d'un "
             "meme prompt ne sont pas independantes ; l'accord de cellules "
             "est descriptif et ne porte aucun test.")
    L.append("- Le detournement de tache n'est pas l'ASR officiel d'OPI : "
             "il mesure le changement de tache, pas la justesse de la "
             "reponse a la tache injectee, pour laquelle OPI ne fournit "
             "pas d'etiquette de verite.")
    L.append("- Aucune de ces quantites n'est calculable a partir d'un "
             "seul prompt sans connaitre la frontiere des regions. Comme "
             "caracteristique de detection, elles supposent la "
             "segmentation connue.")
    L.append("")

    txt = "\n".join(L)
    out = os.path.join(args.root, "RAPPORT.md")
    with open(out, "w", encoding="utf-8") as fh:
        fh.write(txt)
    print(txt)
    print(f"\n--- ecrit : {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
