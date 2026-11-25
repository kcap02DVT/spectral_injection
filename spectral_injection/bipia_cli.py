"""BIPIA email-QA pipeline.

    python -m spectral_injection.bipia_cli --model tinyllama --n-pairs 20

Downloads the BIPIA email and text-attack files on first use, builds paired
prompts (same email, clean and injected), runs both through the model, and
writes per-pair metrics plus per-layer AUROC.

Protocol note: use ``--split train`` to choose a layer, then ``--split test``
to report it. The attack categories are disjoint between the two splits, so
that sequence is a genuine out-of-distribution evaluation rather than a
layer picked on the numbers being reported.
"""

import argparse
import os
import sys
from typing import List

import numpy as np

from .bipia import POSITIONS, describe, load_pairs
from .controls import control_segments
from .cli import (
    _fiedler_drop_head, _graph_view, _head_graph_view, _write_csv,
)
from .config import RunConfig
from .evaluate import (
    best_layer, evaluate_pairs, summarise_by_category, summarise_by_layer,
    summarise_by_position, summarise_per_head,
)
from .metrics import per_head_available
from .report import write_report
from .viz import figure_head_grid, figure_layer_grid, figure_pair


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="BIPIA email-QA spectral analysis.")
    p.add_argument("--model", default="tinyllama")
    p.add_argument("--split", default="test", choices=["train", "test"])
    p.add_argument("--n-pairs", type=int, default=20)
    p.add_argument("--positions", nargs="+", default=list(POSITIONS),
                   choices=list(POSITIONS))
    p.add_argument("--max-context-chars", type=int, default=1200)
    p.add_argument("--attack-source", default="bipia",
                   choices=["bipia", "neuralchemy"],
                   help="origine des attaques : fichiers text_attack de BIPIA, "
                        "ou neuralchemy/Prompt-injection-dataset (config core). "
                        "Les emails hôtes viennent toujours de BIPIA")
    p.add_argument("--attack-category", default=None,
                   help="ne garder qu'une catégorie d'attaque de la source ; "
                        "défaut neuralchemy : direct_injection")
    p.add_argument("--attack-max-chars", type=int, default=None,
                   help="écarter les attaques plus longues ; recommandé avec "
                        "neuralchemy (10 à 7009 caractères)")
    p.add_argument("--attack-severity", nargs="+", default=None,
                   choices=["low", "medium", "high", "critical"],
                   help="neuralchemy uniquement : ne garder que ces niveaux "
                        "de la colonne severity (ex. --attack-severity low "
                        "critical)")
    p.add_argument("--max-tokens", type=int, default=768)
    p.add_argument("--layer", type=int, default=None,
                   help="layer for the figure and the breakdowns (default: middle)")
    p.add_argument("--layers", nargs="+", type=int, default=None,
                   help="restrict metrics to these layers (default: all)")

    p.add_argument("--device", default="auto")
    p.add_argument("--dtype", default="float32", choices=["float32", "float16"])
    p.add_argument("--normalization", default="sym", choices=["sym", "rw", "none"])
    p.add_argument("--drop-sink", action="store_true",
                   help="drop the highest-mass node (off by default)")
    p.add_argument("--drop-template", action="store_true",
                   help="no-op in raw-text mode; kept for older result files")
    p.add_argument("--color-by", default="role", choices=["role", "cluster"])
    p.add_argument("--top-k-edges", type=int, default=None,
                   help="cap on the number of edges drawn; omitted = draw all")
    p.add_argument("--edge-scale", default="log",
                   choices=["log", "linear", "rank"],
                   help="how edge weight maps to opacity")
    p.add_argument("--edge-gamma", type=float, default=3.0,
                   help="curve of the alpha/width ramp")
    p.add_argument("--edge-ink-budget", type=float, default=250.0,
                   help="total opacity summed over all edges; 0 disables")
    p.add_argument("--edge-curvature", type=float, default=0.75,
                   help="bow the chords towards the centre; 0 = straight")
    p.add_argument("--edge-outward", type=float, default=0.55,
                   help="how far short-range chords bulge outside the ring")
    p.add_argument("--edge-outward-span", type=float, default=6.0,
                   help="span in tokens at which the outward bulge peaks")
    p.add_argument("--edge-color-by", default="family",
                   choices=["family", "weight"],
                   help="'family' gives cross edges their own hue")
    p.add_argument("--node-gap-frac", type=float, default=0.04,
                   help="blank arc opened at each role boundary")

    p.add_argument("--per-head", action="store_true",
                   help="also collect per-head metrics (needs spectral-trust) "
                        "and report which heads separate the two conditions")
    p.add_argument("--per-head-top", type=int, default=15,
                   help="how many (layer, head, metric) triples to print")
    p.add_argument("--head-grid", action="store_true",
                   help="avec --per-head : dessine une grille des têtes "
                        "(une vignette d'adjacence par tête) sur chaque "
                        "couche tracée, dans les deux conditions")
    p.add_argument("--head-grid-ncols", type=int, default=8)
    p.add_argument("--head-grid-panel", default="adjacency",
                   choices=["adjacency", "graph", "both"],
                   help="type de vignette : matrice d'adjacence, graphe "
                        "circulaire, ou les deux grilles")
    p.add_argument("--head-grid-top-k", type=int, default=None,
                   help="plafond d'arêtes par tête pour --head-grid-panel "
                        "graph ; omis = toutes (lent : ~54000 arêtes x 32)")
    p.add_argument("--head-grid-conditions", default="injected",
                   choices=["injected", "both"],
                   help="par défaut seule la condition injectée est tracée ; "
                        "'both' ajoute la bénigne, qui sert de contrôle")
    p.add_argument("--head-top", type=int, default=8,
                   help="nombre de têtes tracées, classées par separation ; "
                        "0 = toutes")
    p.add_argument("--head-min", type=float, default=None,
                   help="ne garder que les têtes dont la separation dépasse "
                        "ce seuil (1.0 = présence d'un bicluster)")

    p.add_argument("--figures", default="first",
                   choices=["first", "all", "none"],
                   help="'first' (défaut) : une figure pour la paire 0 ; "
                        "'all' : une figure par paire, dans figures/ ; "
                        "'none' : aucune")
    p.add_argument("--compare", default="paired",
                   choices=["paired", "triptych", "control"],
                   help="'paired' (défaut) : bénin | injecté ; 'triptych' : "
                        "bénin | injecté | injection seule (l'attaque sans "
                        "email ni tâche) ; 'control' : bénin | injecté | "
                        "contrôle inséré de même longueur en tokens, avec "
                        "|injecté − bénin| et |contrôle − bénin| côte à côte. "
                        "Figures uniquement : une passe du modèle en plus par "
                        "paire tracée, métriques et CSV d'évaluation inchangés")
    p.add_argument("--control", default="email_text", choices=["email_text"],
                   help="texte de contrôle pour --compare control. "
                        "'email_text' : phrases consécutives du corps (après "
                        "CONTENT:) d'un AUTRE email du split, sans suite de 6 "
                        "mots commune avec l'email hôte, même nombre de "
                        "tokens que l'attaque (tokenizer du modèle, "
                        "troncature au dernier mot entier si besoin)")
    p.add_argument("--figure-pair", type=int, default=0,
                   help="index de la paire tracée quand --figures first")
    p.add_argument("--figure-pairs", nargs="+", type=int, default=None,
                   help="indices précis des paires à tracer, ex: 0 1 2. "
                        "L'évaluation porte quand même sur --n-pairs paires, "
                        "donc les CSV restent ceux de la campagne complète")

    p.add_argument("--grid", action="store_true",
                   help="figure détaillée complète pour CHAQUE couche, plus "
                        "la grille d'adjacence de survol. Sans lui, seule la "
                        "couche --layer est tracée")
    p.add_argument("--layer-select", default="none",
                   choices=["none", "separation", "modularity", "conductance"],
                   help="choisir les couches tracées d'après les statistiques "
                        "de coupe plutôt que par index : 'separation' garde "
                        "celles où les deux familles forment deux blocs")
    p.add_argument("--layer-top", type=int, default=3,
                   help="nombre de couches retenues par --layer-select")
    p.add_argument("--layer-min", type=float, default=None,
                   help="seuil minimal sur le critère (ex: 1.0 pour "
                        "separation : en dessous, pas de bicluster)")
    p.add_argument("--grid-panel", default="adjacency",
                   choices=["adjacency", "graph"])
    p.add_argument("--grid-ncols", type=int, default=6)
    p.add_argument("--cache-dir", default="data/bipia")
    p.add_argument("--offline", action="store_true")
    p.add_argument("--out", default="results/bipia")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--quiet", action="store_true")
    p.add_argument("--no-report", action="store_true",
                   help="ne pas écrire le rapport LaTeX du run")
    return p


def _print_table(title: str, rows, keys) -> None:
    if not rows:
        return
    print(f"\n{title}")
    print("-" * 78)
    print("".join(f"{k:>18}" if i else f"{k:<24}"
                  for i, k in enumerate(keys)))
    for row in rows:
        cells = []
        for i, k in enumerate(keys):
            v = row.get(k, float("nan"))
            text = f"{v:.4f}" if isinstance(v, float) else str(v)
            cells.append(f"{text:>18}" if i else f"{text:<24}")
        print("".join(cells))


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)

    config = RunConfig(
        model=args.model, device=args.device, dtype=args.dtype,
        normalization=args.normalization,
        drop_sink=args.drop_sink, drop_template=args.drop_template,
        max_tokens=args.max_tokens, out_dir=args.out,
        color_by=args.color_by, seed=args.seed,
        top_k_edges=args.top_k_edges, edge_scale=args.edge_scale,
        edge_gamma=args.edge_gamma,
        edge_ink_budget=args.edge_ink_budget or None,
        edge_curvature=args.edge_curvature, edge_outward=args.edge_outward,
        edge_outward_span=args.edge_outward_span,
        edge_color_by=args.edge_color_by, node_gap_frac=args.node_gap_frac,
        per_head=args.per_head,
    )
    out_dir = os.path.join(args.out, config.model_slug, args.split)
    os.makedirs(out_dir, exist_ok=True)

    # Checked before any model is loaded: discovering this after an hour of
    # forward passes costs the whole per-head half of the campaign.
    if args.per_head:
        reason = per_head_available()
        if reason:
            print("=" * 72, file=sys.stderr)
            print("--per-head demandé mais INDISPONIBLE — le run continuera "
                  "sans.", file=sys.stderr)
            print(f"  {reason}", file=sys.stderr)
            print("  Activez le venv du projet :", file=sys.stderr)
            print("    .\\spectral_injection\\env\\Scripts\\Activate.ps1",
                  file=sys.stderr)
            print("  Ctrl+C maintenant pour ne pas perdre le calcul.",
                  file=sys.stderr)
            print("=" * 72, file=sys.stderr)

    print(f"BIPIA email-QA | split={args.split} | modèle={config.model_id}")
    pairs = load_pairs(
        split=args.split, n_pairs=args.n_pairs, positions=args.positions,
        max_context_chars=args.max_context_chars, cache_dir=args.cache_dir,
        offline=args.offline, seed=args.seed,
        attack_source=args.attack_source,
        attack_category=args.attack_category,
        attack_max_chars=args.attack_max_chars,
        attack_severity=args.attack_severity,
    )
    print(describe(pairs))
    longueurs = [len(p.attack) for p in pairs]
    print(f"source {args.attack_source}"
          + (f" · catégorie {args.attack_category}" if args.attack_category
             else "")
          + (f" · sévérité {'/'.join(args.attack_severity)}"
             if args.attack_severity else "")
          + f" · longueur des attaques : médiane "
            f"{sorted(longueurs)[len(longueurs) // 2]} car., "
            f"max {max(longueurs)}")

    from .runner import AttentionRunner       # torch imported late

    runner = AttentionRunner(config)
    layer = args.layer if args.layer is not None else runner.middle_layer()
    print(f"Couches : {runner.n_layers} | couche analysée : {layer}\n")

    control_pool = None
    control_log = os.path.join(out_dir, "controls.csv")
    if args.compare == "control":
        from .bipia import _load_emails, ensure_data
        from .controls import EmailTextControl

        emails_split = _load_emails(ensure_data(
            args.cache_dir, offline=args.offline)[f"email_{args.split}"])
        control_pool = EmailTextControl(emails_split, runner.tokenizer,
                                        seed=args.seed)
        os.makedirs(out_dir, exist_ok=True)
        if os.path.exists(control_log):         # a previous run's draws
            os.remove(control_log)
        print(f"contrôle {args.control} : {len(control_pool.windows)} "
              f"fenêtres de phrases (emails {args.split}) ; "
              f"tirages consignés dans controls.csv\n")

    # Figures are drawn inside the evaluation loop, where the attentions are
    # already in hand: re-running the model afterwards would double the cost.
    fig_dir = os.path.join(out_dir, "figures")
    written: List[str] = []

    # --figure-pairs wins over --figures: it names the pairs explicitly while
    # the evaluation still runs on all of them, so the CSVs stay those of the
    # full campaign.
    wanted_pairs = (set(args.figure_pairs)
                    if args.figure_pairs is not None else None)
    if wanted_pairs is not None:
        out_of_range = sorted(i for i in wanted_pairs if not 0 <= i < len(pairs))
        if out_of_range:
            print(f"--figure-pairs: indices hors plage ignorés "
                  f"{out_of_range} (0..{len(pairs) - 1})", file=sys.stderr)
            wanted_pairs -= set(out_of_range)
        if not wanted_pairs:
            print("--figure-pairs: aucun indice valide, aucune figure.",
                  file=sys.stderr)

    # Les couches dessinées sont toujours un sous-ensemble des couches
    # calculées : une figure dont les métriques ne figurent dans aucun CSV
    # n'est pas vérifiable. --layers restreint le calcul, donc il restreint
    # aussi le dessin ; --grid dessine tout ce qui a été calculé.
    computed = (list(args.layers) if args.layers is not None
                else list(range(runner.n_layers)))
    if args.grid or args.layers is not None:
        fig_layers = computed
    else:
        fig_layers = [layer]

    if layer not in computed:
        substitute = computed[len(computed) // 2]
        print(f"--layer {layer} n'est pas dans les couches calculées "
              f"{computed} ; les ventilations porteront sur L{substitute}.",
              file=sys.stderr)
        layer = substitute

    def select_layers(table_i):
        """Layers where the injected graph actually splits in two.

        Rather than drawing every layer and letting the reader hunt, this
        ranks them by a cut statistic computed on that pair and keeps the top
        few. ``separation`` is the direct expression of the hypothesis:
        sqrt(intra_benign * intra_injection) / cross, so above 1 the two
        families hold together more than they talk to each other.
        ``conductance`` runs the other way -- a sharper cut is a smaller
        number -- so it is ranked ascending.
        """
        key = args.layer_select
        rows = [r for r in table_i if np.isfinite(r.get(key, np.nan))]
        if args.layer_min is not None:
            rows = [r for r in rows
                    if (r[key] <= args.layer_min if key == "conductance"
                        else r[key] >= args.layer_min)]
        if not rows:
            return []
        rows.sort(key=lambda r: r[key], reverse=(key != "conductance"))
        chosen = rows[:max(args.layer_top, 1)]
        return [(int(r["layer"]), float(r[key])) for r in chosen]

    def draw(idx, pair, lab_b, attn_b, lab_i, attn_i, table_b, table_i):
        if wanted_pairs is not None:
            if idx not in wanted_pairs:
                return
        elif args.figures == "none":
            return
        elif args.figures == "first" and idx != args.figure_pair:
            return
        # As soon as more than one pair is drawn the filename has to carry the
        # pair index: two pairs sharing an insertion position would otherwise
        # write to the same name and silently overwrite each other.
        multi = wanted_pairs is not None or args.figures == "all"
        target = fig_dir if multi else out_dir
        os.makedirs(target, exist_ok=True)
        prefix = f"pair{idx:03d}" if multi else "figure"

        # --grid: the full detailed figure for every layer. --layer alone:
        # that one layer. --layers, when given, restricts both.
        if args.layer_select != "none":
            picked = select_layers(table_i)
            if not picked and not args.quiet:
                print(f"  [figure] paire {idx} : aucune couche ne passe le "
                      f"critère {args.layer_select}, rien tracé")
        else:
            picked = [(lay, None) for lay in fig_layers]

        # --compare triptych: the attack alone, run once per drawn pair and
        # reused for every layer. Never enters the metrics -- see
        # BipiaPair.isolated_segments for why it cannot.
        lab_s = attn_s = None
        if args.compare == "triptych" and picked:
            lab_s, attn_s = runner.analyse_segments(pair.isolated_segments)

        # --compare control: same slot as the attack, same token count, text
        # from another email. Drawn once per pair, logged for traceability.
        lab_c = attn_c = ctl = ctl_segments = None
        if control_pool is not None and picked:
            import csv
            from .prompts import build_labelled_prompt_from_segments

            def in_prompt(text, _pair=pair):
                lab = build_labelled_prompt_from_segments(
                    runner.tokenizer, control_segments(_pair, text),
                    system_prompt=config.system_prompt)
                return lab.count("injection"), len(lab)

            # Same criteria as specificity_cli, so a figure shows the very
            # control the statistics were computed on.
            ctl = control_pool.choose(
                pair, idx, in_prompt=in_prompt,
                n_target=(lab_i.count("injection"), len(lab_i)),
                match_chars=True)
            ctl_segments = control_segments(pair, ctl.text)
            lab_c, attn_c = runner.analyse_segments(ctl_segments)
            new = not os.path.exists(control_log)
            with open(control_log, "w" if new else "a", newline="",
                      encoding="utf-8") as fh:
                w = csv.writer(fh)
                if new:
                    w.writerow(["pair", "position", "host_email", "donor_email",
                                "tokens_attack", "tokens_control",
                                "tokens_inserted_injected",
                                "tokens_inserted_control", "truncated",
                                "candidates", "attack", "control"])
                # host index in the same numbering as donor_email (the full
                # split file), not example_id, which counts filtered emails.
                w.writerow([idx, pair.position,
                            control_pool.contexts.index(pair.context),
                            ctl.donor_index, ctl.n_tokens_attack,
                            ctl.n_tokens_control, lab_i.count("injection"),
                            lab_c.count("injection"), int(ctl.truncated),
                            ctl.n_candidates, pair.attack, ctl.text])

        for rank, (lay, score) in enumerate(picked, start=1):
            path = os.path.join(
                target, f"{prefix}_L{lay}_{pair.position}.png")
            why = (f" · {args.layer_select}={score:.2f} "
                   f"(rang {rank}/{len(picked)})" if score is not None else "")
            figure_pair(
                _graph_view(attn_b[lay], lab_b, config),
                _graph_view(attn_i[lay], lab_i, config),
                config.model_slug, lay, runner.n_layers, config, path,
                benign_segments=pair.benign_segments,
                injected_segments=pair.injected_segments,
                subtitle=f"paire {idx} · email #{pair.example_id} · "
                         f"insertion {pair.position} · "
                         f"{pair.attack_category}{why}",
                isolated=(_graph_view(attn_s[lay], lab_s, config)
                          if attn_s is not None else None),
                isolated_segments=pair.isolated_segments,
                control=(_graph_view(attn_c[lay], lab_c, config)
                         if attn_c is not None else None),
                control_segments=ctl_segments,
                control_note=ctl.label() if ctl is not None else "",
            )
            written.append(path)

            # Avec --per-head : la même figure, mais sur la tête dont la valeur
            # de Fiedler chute le plus entre les deux conditions. La figure
            # agrégée ci-dessus est conservée : les deux se lisent ensemble,
            # l'une montrant ce que la moyenne dilue, l'autre ce qu'une tête
            # isolée voit.
            if args.per_head:
                pick = _fiedler_drop_head(attn_b, lab_b, attn_i, lab_i, lay,
                                          config)
                if pick is None:
                    print(f"  [figure] paire {idx} L{lay} : aucune tête "
                          f"exploitable", file=sys.stderr)
                else:
                    h, vb, vi, drop = pick
                    hpath = os.path.join(
                        target,
                        f"{prefix}_L{lay}_{pair.position}_tete{h:02d}.png")
                    figure_pair(
                        _head_graph_view(attn_b[lay], lab_b, config, h),
                        _head_graph_view(attn_i[lay], lab_i, config, h),
                        config.model_slug, lay, runner.n_layers, config, hpath,
                        benign_segments=pair.benign_segments,
                        injected_segments=pair.injected_segments,
                        subtitle=f"paire {idx} · tête {h} seule · "
                                 f"$\\lambda_2$ {vb:.4f} → {vi:.4f} "
                                 f"({-drop:+.1%}) · insertion {pair.position}",
                        isolated=(_head_graph_view(attn_s[lay], lab_s, config, h)
                                  if attn_s is not None else None),
                        isolated_segments=pair.isolated_segments,
                        control=(_head_graph_view(attn_c[lay], lab_c, config, h)
                                 if attn_c is not None else None),
                        control_segments=ctl_segments,
                        control_note=ctl.label() if ctl is not None else "",
                    )
                    written.append(hpath)

            if args.per_head and args.head_grid:
                panels = (["adjacency", "graph"]
                          if args.head_grid_panel == "both"
                          else [args.head_grid_panel])
                conds = [("injected", attn_i, lab_i)]
                if args.head_grid_conditions == "both":
                    conds.insert(0, ("benign", attn_b, lab_b))
                for label, attns, lab in conds:
                    for pan in panels:
                        hpath = os.path.join(
                            target,
                            f"{prefix}_L{lay}_{pair.position}_heads_"
                            f"{pan}_{label}.png")
                        figure_head_grid(
                            attns[lay], lab.roles, config, hpath, lay,
                            config.model_slug,
                            condition=("bénin" if label == "benign"
                                       else "avec injection"),
                            ncols=args.head_grid_ncols, panel=pan,
                            tokens=lab.tokens,
                            top_k_edges=args.head_grid_top_k,
                            top=(args.head_top or None),
                            min_score=args.head_min)
                        written.append(hpath)

        if args.grid:
            # The adjacency overview stays: it is the cheap view that answers
            # "at what depth does the structure appear?" at a glance, which no
            # single detailed figure can.
            for label, attns, lab in (("benign", attn_b, lab_b),
                                      ("injected", attn_i, lab_i)):
                views = {i: _graph_view(a, lab, config)
                         for i, a in enumerate(attns)}
                gpath = os.path.join(
                    target,
                    f"{prefix}_{pair.position}_grid_{label}_"
                    f"{args.grid_panel}.png")
                figure_layer_grid(
                    views, config.model_slug, config, gpath,
                    panel=args.grid_panel, ncols=args.grid_ncols,
                    condition=("bénin" if label == "benign"
                               else "avec injection"))
                written.append(gpath)

    if (wanted_pairs or args.figures != "none") and not args.quiet:
        if wanted_pairs is not None:
            n_drawn = len(wanted_pairs)
        else:
            n_drawn = len(pairs) if args.figures == "all" else 1
        n_lay = (args.layer_top if args.layer_select != "none"
                 else len(fig_layers))
        per_pair = n_lay + (2 if args.grid else 0)
        total = n_drawn * per_pair
        how = (f"les {n_lay} meilleures sur {args.layer_select}"
               if args.layer_select != "none" else f"{n_lay} couche(s)")
        print(f"  figures : {how} x {n_drawn} paire(s)"
              + (" + 2 grilles/paire" if args.grid else "")
              + f" = au plus {total} PNG (~{total * 1.5:.0f} Mo)")
        if total > 100:
            print("  volume important : --layers ou --figures first "
                  "restreignent la sortie.")
        print()

    per_head = None
    ranking = None
    if args.per_head:
        rows, per_head = evaluate_pairs(runner, pairs, config,
                                        layers=args.layers,
                                        verbose=not args.quiet,
                                        return_per_head=True,
                                        figure_hook=draw)
    else:
        rows = evaluate_pairs(runner, pairs, config, layers=args.layers,
                              verbose=not args.quiet, figure_hook=draw)
    if not rows:
        print("Aucune paire exploitable.", file=sys.stderr)
        return 1

    _write_csv(rows, os.path.join(out_dir, "per_pair_metrics.csv"))
    summary = summarise_by_layer(rows)
    _write_csv(summary, os.path.join(out_dir, "summary_by_layer.csv"))

    _print_table(
        "AUROC par couche (bénin vs injecté, score sans label)",
        summary,
        ["layer", "auroc_cut_conductance", "auroc_fiedler_value",
         "mean_separation", "mean_accuracy"],
    )

    top = best_layer(summary)
    if top:
        print(f"\nMeilleure couche sur cut_conductance : L{top['layer']} "
              f"— AUROC={top['auroc_cut_conductance']:.4f} "
              f"(direction={top['dir_cut_conductance']:+.0f}, "
              f"-1 = coupe plus nette quand injecté)")
        if args.split == "test" and args.layer is None:
            print("  Attention : couche choisie sur les lignes mêmes qui la "
                  "rapportent. Pour le papier, choisir sur --split train.")

    position_rows = summarise_by_position(rows, layer)
    category_rows = summarise_by_category(rows, layer)
    _print_table(
        f"Par position d'insertion (couche {layer})", position_rows,
        ["position", "n", "mean_accuracy", "mean_separation", "mean_conductance"],
    )
    _print_table(
        f"Catégories d'attaque les plus difficiles (couche {layer})",
        category_rows,
        ["attack_category", "n", "mean_accuracy", "mean_separation"],
    )

    if args.per_head:
        if per_head is None:
            print("\n[per-head] spectral-trust indisponible — rien à exporter.",
                  file=sys.stderr)
        else:
            values = per_head["values"]
            np.save(os.path.join(out_dir, "per_head_values.npy"), values)
            _write_csv(per_head["index"],
                       os.path.join(out_dir, "per_head_index.csv"))
            ranking = summarise_per_head(per_head, top=0)
            _write_csv(ranking, os.path.join(out_dir, "per_head_auroc.csv"))

            n_obs, n_lay, n_head, n_met = values.shape
            _print_table(
                f"Têtes les plus discriminantes ({n_obs} observations)",
                ranking[:args.per_head_top],
                ["metric", "layer", "head", "auroc", "direction"],
            )
            top_head = ranking[0]
            top_agg = best_layer(summary)
            print(f"\n  meilleure tête  : L{top_head['layer']} "
                  f"H{top_head['head']} {top_head['metric']} — "
                  f"AUROC={top_head['auroc']:.4f}")
            if top_agg:
                print(f"  meilleur agrégé : L{top_agg['layer']} "
                      f"cut_conductance — "
                      f"AUROC={top_agg['auroc_cut_conductance']:.4f}")

            # The number above is the maximum of n_lay*n_head*n_met AUROCs
            # computed on n_obs points. That maximum is high even when every
            # head is pure noise, so it is not a detection result.
            n_cand = n_lay * n_head * n_met
            print(f"\n  ATTENTION : ce maximum est pris sur {n_cand} candidats "
                  f"({n_lay} couches x {n_head} têtes x {n_met} métriques) "
                  f"évalués sur {n_obs} observations.")
            print("  Le maximum de milliers d'AUROC est élevé même sous "
                  "l'hypothèse nulle. Pour un chiffre publiable : choisir la "
                  "tête sur --split train, la rapporter sur --split test.")
            print(f"\n  per_head_values.npy  {values.shape} "
                  f"[obs, couche, tête, métrique]")
            print(f"  per_head_index.csv   {n_obs} lignes de métadonnées")
            print(f"  per_head_auroc.csv   {len(ranking)} triplets")

    if not args.no_report:
        try:
            report_path = write_report(
                os.path.join(out_dir, "rapport.tex"), config, args, rows,
                summary, layer, runner.n_layers, pairs, position_rows,
                category_rows, per_head=per_head,
                head_ranking=(ranking if args.per_head and per_head else None),
                figures=written)
            print(f"\nRapport -> {os.path.basename(report_path)}")
        except Exception as exc:                          # noqa: BLE001
            # A failed report must not discard an hour of forward passes, but
            # it must be visible: this is the artefact the run exists to leave.
            print(f"\n[rapport] ÉCHEC : {type(exc).__name__}: {exc}",
                  file=sys.stderr)

    print(f"\nSorties -> {out_dir}")
    print(f"  per_pair_metrics.csv   ({len(rows)} lignes)")
    print(f"  summary_by_layer.csv   ({len(summary)} couches)")
    if written:
        print(f"  {len(written)} figure(s)")
        for path in written[:3]:
            print(f"    {os.path.relpath(path, out_dir)}")
        if len(written) > 3:
            print(f"    ... et {len(written) - 3} autres")
    elif args.figures != "none":
        print("  aucune figure écrite (aucune paire exploitable ?)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
