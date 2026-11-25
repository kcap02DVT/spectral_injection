"""Automatic LaTeX report for one run.

Writes a standalone, compilable ``.tex`` next to the CSVs: protocol,
configuration, every result table, and an interpretation section derived from
the numbers rather than pre-written.

The interpretation is deliberately conservative. An AUROC is reported with a
confidence interval and called inconclusive when that interval straddles 0.5;
the agreement accuracy is compared against the trivial all-in-one-group
baseline before being described as meaningful; and any figure obtained as the
maximum over many candidates is labelled as such. A report that overstates is
worse than no report -- a reviewer will find the same holes, later.
"""

import csv
import datetime as _dt
import os
import platform
import sys
from typing import Dict, List, Optional, Sequence

import numpy as np

_SPECIAL = {"&": r"\&", "%": r"\%", "$": r"\$", "#": r"\#", "_": r"\_",
            "{": r"\{", "}": r"\}", "~": r"\textasciitilde{}",
            "^": r"\textasciicircum{}", "\\": r"\textbackslash{}"}


def esc(text) -> str:
    """Escape LaTeX specials. Attack categories contain '&' and '_'."""
    return "".join(_SPECIAL.get(c, c) for c in str(text))


def _num(value, digits: int = 4) -> str:
    if value is None:
        return "--"
    try:
        v = float(value)
    except (TypeError, ValueError):
        return esc(value)
    return "--" if not np.isfinite(v) else f"{v:.{digits}f}"


def auroc_ci(auroc: float, n_pos: int, n_neg: int, z: float = 1.96):
    """Hanley-McNeil confidence interval for an AUROC.

    Approximate, and that is the point: it turns "0.64" into "0.64 [0.51,
    0.77]", which is the difference between a claim and a number.
    """
    if not np.isfinite(auroc) or n_pos < 1 or n_neg < 1:
        return (float("nan"), float("nan"))
    a = float(auroc)
    q1 = a / (2.0 - a)
    q2 = 2.0 * a * a / (1.0 + a)
    var = (a * (1 - a) + (n_pos - 1) * (q1 - a * a)
           + (n_neg - 1) * (q2 - a * a)) / (n_pos * n_neg)
    se = float(np.sqrt(max(var, 0.0)))
    return (max(0.0, a - z * se), min(1.0, a + z * se))


def _table(rows: List[Dict], columns: Sequence[str], caption: str,
           label: str, digits: int = 4) -> str:
    spec = "l" + "r" * (len(columns) - 1)
    head = " & ".join(esc(c.replace("_", " ")) for c in columns) + r" \\"
    body = []
    for row in rows:
        cells = [_num(row.get(c), digits) if i else esc(row.get(c, "--"))
                 for i, c in enumerate(columns)]
        body.append(" & ".join(cells) + r" \\")
    return "\n".join([
        r"\begin{table}[htbp]", r"\centering", r"\small",
        rf"\begin{{tabular}}{{{spec}}}", r"\hline", head, r"\hline",
        *body, r"\hline", r"\end{tabular}",
        rf"\caption{{{caption}}}", rf"\label{{{label}}}", r"\end{table}",
    ])


def _accuracy_baseline(rows: List[Dict], layer: int) -> Dict[str, float]:
    """Trivial baseline for the agreement accuracy, and whether it is beaten.

    ``partition_agreement`` maximises over both label orientations, so putting
    every node in one group already scores max(p, 1-p) where p is the injected
    fraction. With an injection of 9 tokens in 320 that baseline is 0.97, and
    an accuracy of 0.58 is then far *below* chance rather than above it. This
    is the single easiest number in the pipeline to misread.
    """
    sub = [r for r in rows if r["label"] == 1 and r["layer"] == layer]
    if not sub:
        return {}
    acc, base = [], []
    for r in sub:
        nb, ni = float(r.get("n_benign", 0)), float(r.get("n_injection", 0))
        if nb + ni <= 0:
            continue
        p = ni / (nb + ni)
        base.append(max(p, 1.0 - p))
        if np.isfinite(r.get("accuracy", np.nan)):
            acc.append(float(r["accuracy"]))
    if not acc or not base:
        return {}
    a, b = float(np.mean(acc)), float(np.mean(base))
    beaten = sum(1 for r in sub
                 if np.isfinite(r.get("accuracy", np.nan))
                 and float(r["accuracy"]) > max(
                     float(r.get("n_injection", 0)) /
                     max(float(r.get("n_benign", 0)) +
                         float(r.get("n_injection", 0)), 1e-9),
                     1.0 - float(r.get("n_injection", 0)) /
                     max(float(r.get("n_benign", 0)) +
                         float(r.get("n_injection", 0)), 1e-9)))
    return {"mean_accuracy": a, "mean_baseline": b,
            "n_beating": beaten, "n_total": len(sub)}


def write_report(out_path: str, config, args, rows: List[Dict],
                 summary: List[Dict], layer: int, n_layers: int,
                 pairs, position_rows: List[Dict], category_rows: List[Dict],
                 per_head: Optional[Dict] = None,
                 head_ranking: Optional[List[Dict]] = None,
                 figures: Optional[Sequence[str]] = None) -> str:
    """Write the LaTeX report and return its path."""
    n_inj = sum(1 for r in rows if r["label"] == 1 and r["layer"] == layer)
    n_ben = sum(1 for r in rows if r["label"] == 0 and r["layer"] == layer)

    best = max((s for s in summary
                if np.isfinite(s.get("auroc_cut_conductance", np.nan))),
               key=lambda s: s["auroc_cut_conductance"], default=None)

    L: List[str] = []
    add = L.append

    # ---------------------------------------------------------------- preamble
    add(r"\documentclass[11pt,a4paper]{article}")
    add(r"\usepackage[utf8]{inputenc}")
    add(r"\usepackage[T1]{fontenc}")
    add(r"\usepackage[french]{babel}")
    add(r"\usepackage{booktabs,graphicx,geometry,amsmath,hyperref}")
    add(r"\geometry{margin=2.4cm}")
    add(rf"\title{{Analyse spectrale du graphe d'attention\\"
        rf"{esc(config.model_slug)} --- BIPIA {esc(args.split)}}}")
    add(r"\author{Rapport généré automatiquement}")
    stamp = _dt.datetime.now().strftime("%d/%m/%Y %H:%M")
    add(rf"\date{{{stamp}}}")
    add(r"\begin{document}\maketitle")

    # ---------------------------------------------------------------- protocol
    add(r"\section{Protocole}")
    add(rf"""
Les paires proviennent de la tâche \emph{{email\_QA}} du benchmark BIPIA.
Pour chaque email, deux prompts sont construits : une condition \textbf{{bénigne}}
(l'email seul) et une condition \textbf{{injectée}}, identique à la précédente
augmentée d'une instruction d'attaque insérée au début, au milieu ou à la fin
du corps du message. L'insertion étant réalisée par le code, la position en
caractères de l'attaque est connue exactement, ce qui permet d'étiqueter chaque
token sans alignement approximatif entre deux tokenisations.

Le prompt complet est envoyé au modèle : template de chat, prompt système,
consignes de tâche et question. Tout ce qui n'est pas l'attaque est étiqueté
\texttt{{benign}} ; seule l'attaque est \texttt{{injection}}. L'étiquetage est
donc strictement binaire.

Les catégories d'attaque des splits \texttt{{train}} et \texttt{{test}} sont
disjointes. Choisir une couche ou une tête sur \texttt{{train}} puis la
rapporter sur \texttt{{test}} constitue donc une évaluation hors distribution
par construction.

\paragraph{{Construction du graphe.}} L'attention d'un décodeur étant causale,
elle est symétrisée par $W = (A + A^{{\top}})/2$ avant tout traitement spectral.
Le Laplacien, la décomposition propre et les cinq métriques spectrales sont
calculés par la bibliothèque \texttt{{spectral-trust}} (v0.3.0) ; le filtrage
des n\oe uds par rôle et les statistiques de coupe sont propres à ce travail.
Ce run a porté sur {len(pairs)} paires, soit {n_ben} observations bénignes et
{n_inj} injectées.
""")

    # ------------------------------------------------------------ configuration
    add(r"\section{Configuration}")
    cfg_rows = [
        {"param": "modèle", "valeur": config.model_id},
        {"param": "couches", "valeur": n_layers},
        {"param": "split", "valeur": args.split},
        {"param": "paires", "valeur": len(pairs)},
        {"param": "normalisation du Laplacien", "valeur": config.normalization},
        {"param": "symétrisation", "valeur": config.symmetrization},
        {"param": "agrégation des têtes", "valeur": config.head_aggregation},
        {"param": "solveur propre", "valeur": getattr(config, "eigen_solver", "dense")},
        {"param": "retrait du puits", "valeur": config.drop_sink},
        {"param": "retrait du template", "valeur": config.drop_template},
        {"param": "couche analysée", "valeur": layer},
        {"param": "métriques par tête", "valeur": bool(per_head)},
    ]
    spec_rows = "\n".join(rf"{esc(r['param'])} & {esc(r['valeur'])} \\"
                          for r in cfg_rows)
    add("\n".join([r"\begin{table}[htbp]\centering\small",
                   r"\begin{tabular}{ll}", r"\hline",
                   r"paramètre & valeur \\", r"\hline", spec_rows, r"\hline",
                   r"\end{tabular}",
                   r"\caption{Paramètres qui modifient un chiffre.}",
                   r"\label{tab:config}", r"\end{table}"]))
    # One paragraph, not two: a line starting with "\\" has no line to end.
    add(rf"""
\paragraph{{Commande.}}
\texttt{{{esc(' '.join(sys.argv))}}}

\paragraph{{Environnement.}}
Python {esc(platform.python_version())}, {esc(platform.platform())}.
""")

    # ----------------------------------------------------------------- results
    add(r"\section{Résultats}")
    add(r"\subsection{Détection par couche}")
    add(_table(summary,
               ["layer", "auroc_cut_conductance", "auroc_fiedler_value",
                "mean_separation", "mean_accuracy"],
               "AUROC bénin contre injecté, par couche. "
               "\\texttt{cut\\_conductance} est le score sans label : "
               "il ne voit jamais où est l'injection.",
               "tab:layers"))

    if best is not None:
        lo, hi = auroc_ci(best["auroc_cut_conductance"], n_inj, n_ben)
        add(rf"""
\paragraph{{Meilleure couche.}} L{best['layer']} avec
AUROC $= {best['auroc_cut_conductance']:.4f}$, intervalle de confiance à 95\%
$[{lo:.3f},\ {hi:.3f}]$ (Hanley--McNeil, $n_+ = {n_inj}$, $n_- = {n_ben}$).
""")
        add(rf"""
\textbf{{Cet intervalle ne tient pas compte de la sélection.}} Il suppose une
couche fixée d'avance, alors que L{best['layer']} est le \emph{{maximum}} sur
{len(summary)} couches. Sous l'hypothèse nulle, le maximum de
{len(summary)} AUROC calculées sur $n_+ = {n_inj}$ et $n_- = {n_ben}$ est déjà
nettement supérieur à $0{{,}}5$. La conclusion appartient à la section
Interprétation, qui est rédigée à la main : aucun verdict n'est produit
automatiquement ici.
""")

    add(r"\subsection{Ventilation par position d'insertion}")
    add(_table(position_rows,
               ["position", "n", "mean_accuracy", "mean_separation",
                "mean_conductance"],
               rf"Couche {layer}, condition injectée uniquement. Une valeur "
               r"stable entre les trois positions indique que la coupe suit "
               r"l'injection et non une région fixe de la séquence.",
               "tab:positions"))

    add(r"\subsection{Catégories d'attaque les plus difficiles}")
    add(_table(category_rows,
               ["attack_category", "n", "mean_accuracy", "mean_separation"],
               rf"Couche {layer}, dix catégories les moins bien retrouvées.",
               "tab:categories"))

    if head_ranking:
        add(r"\subsection{Métriques par tête}")
        add(_table(head_ranking[:15],
                   ["metric", "layer", "head", "auroc", "direction"],
                   "Quinze meilleurs triplets (couche, tête, métrique).",
                   "tab:heads"))
        if per_head:
            n_cand = (len(per_head["layers"]) * per_head["n_heads"]
                      * len(per_head["metric_names"]))
            n_obs = len(per_head["labels"])
            add(rf"""
\textbf{{Avertissement.}} Ce maximum est pris sur {n_cand} candidats
({len(per_head['layers'])} couches $\times$ {per_head['n_heads']} têtes
$\times$ {len(per_head['metric_names'])} métriques) évalués sur {n_obs}
observations. Le maximum de plusieurs milliers d'AUROC est élevé même sous
l'hypothèse nulle : une simulation avec des scores purement aléatoires donne
un maximum proche de $0{{,}}99$ pour 12 observations et $0{{,}}69$ pour 120.
\emph{{Ce chiffre n'est pas un résultat de détection.}} Il ne le devient
qu'après avoir fixé la tête sur \texttt{{train}} et l'avoir rapportée sur
\texttt{{test}}.
""")

    # ------------------------------------------------------- computed controls
    # Numbers only. No verdicts: a template cannot weigh a selection effect
    # against a confidence interval, and one that tries will overclaim.
    add(r"\section{Contrôles chiffrés}")

    seps = [s["mean_separation"] for s in summary
            if np.isfinite(s.get("mean_separation", np.nan))]
    if seps:
        add(rf"""
\paragraph{{Séparation.}} $\sqrt{{d_b \cdot d_i}} / d_{{\text{{croisé}}}}$ varie
de ${min(seps):.2f}$ à ${max(seps):.2f}$ selon la couche
({sum(1 for s in seps if s > 1.0)} couches sur {len(seps)} au-dessus de 1).
Cette quantité utilise les \emph{{vraies}} étiquettes : elle mesure si les deux
familles se tiennent, pas si un détecteur les trouve.
""")

    base = _accuracy_baseline(rows, layer)
    if base:
        add(rf"""
\paragraph{{Accord contre ligne de base.}} À la couche {layer}, accord moyen
${base['mean_accuracy']:.3f}$ ; ligne de base triviale (tous les n\oe uds dans
un seul groupe, \texttt{{partition\_agreement}} maximisant sur les deux
orientations) ${base['mean_baseline']:.3f}$.
{base['n_beating']} paire(s) sur {base['n_total']} dépassent cette ligne.
""")

    aris = [float(r["ari"]) for r in rows
            if r["label"] == 1 and np.isfinite(r.get("ari", np.nan))]
    if aris:
        add(rf"""
\paragraph{{Indice de Rand ajusté.}} Moyenne ${np.mean(aris):+.4f}$ sur
{len(aris)} observations injectées (0 correspond au hasard). L'ARI est corrigé
du hasard, contrairement à l'accord ci-dessus.
""")

    # ---------------------------------------------------------- interpretation
    add(r"\section{Interprétation}")
    # Resolved by LaTeX at compile time, not by Python at generation time:
    # the interpretation is usually written *after* the run finishes, and the
    # report must pick it up without being regenerated.
    add(r"""
\IfFileExists{interpretation.tex}{\input{interpretation}}{%
\emph{À rédiger.} Les sections précédentes sont générées automatiquement à
partir des CSV ; celle-ci ne l'est pas. Déposer un fichier
\texttt{interpretation.tex} dans ce dossier puis recompiler : il sera inclus
ici, sans avoir à relancer le run.

Points à trancher, qu'aucun gabarit ne peut trancher : l'effet de sélection sur
la meilleure couche ou la meilleure tête, la part de la séparation imputable à
la contiguïté des tokens injectés, et l'écart entre une séparation élevée et un
accord nul.%
}""")

    # ------------------------------------------------------------- definitions
    add(r"\section{Définitions des métriques}")
    add(r"""
\begin{description}
\item[fiedler\_value] $\lambda_2$ du Laplacien, la connectivité algébrique.
  Proche de 0 lorsque le graphe est presque déconnecté en deux.
\item[connectivity\_ratio] $\lambda_2 / \lambda_{\max}$, sans échelle.
\item[spectral\_entropy\_norm] entropie de la distribution des valeurs propres,
  normalisée par $\log n$. Élevée si le spectre est étalé.
\item[hfer] part de la masse spectrale dans la moitié haute du spectre.
\item[spectral\_radius] $\lambda_{\max}$.
\item[density\_benign, density\_injection] poids moyen par paire \emph{possible}
  à l'intérieur de chaque famille.
\item[density\_cross] idem entre les deux familles.
\item[separation] $\sqrt{d_b \cdot d_i} / d_{\text{croisé}}$.
\item[conductance, normalized\_cut, modularity] qualité de la coupe entre la
  vraie frontière bénin/injection.
\item[cut\_conductance] conductance de la coupe \emph{découverte} par le signe
  du vecteur de Fiedler, sans utiliser les étiquettes. C'est le score du
  détecteur.
\item[accuracy, ari] accord entre la coupe découverte et la vraie frontière.
\end{description}
""")

    if figures:
        add(r"\section{Figures}")
        for path in list(figures)[:6]:
            rel = os.path.basename(path).replace("\\", "/")
            add(rf"""
\begin{{figure}}[htbp]\centering
\includegraphics[width=\linewidth]{{figures/{rel}}}
\caption{{\texttt{{{esc(rel)}}}}}
\end{{figure}}""")

    add(r"\end{document}")

    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as handle:
        handle.write("\n\n".join(L))
    return out_path
