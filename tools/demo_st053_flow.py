"""Les vues publiees de 0.5.3 sur une vraie paire OPI benin / injecte.

La bibliotheque met la vue ``flow`` en avant precisement sur une injection
de prompt. Ce script la reproduit sur NOTRE corpus apparie, en passant les
identifiants de tokens deja formates -- pas du texte -- pour que le prompt
mesure soit EXACTEMENT celui des campagnes (meme gabarit de chat, aucun
message systeme ajoute, aucune retokenisation).

Rend, pour une paire :
  flow.html        les deux bras superposes, un tube par token a travers
                   les couches, colore par son cote du partage de Fiedler
  heatmap.svg      la carte couches x tetes de la valeur de Fiedler
  spectre.svg      le spectre d'une couche declaree
  metriques.csv    toutes les metriques du cache, par couche et par tete

AUCUNE conclusion n'est tiree d'une paire unique : c'est une illustration
et un essai d'integration, pas une mesure.

    python tools/demo_st053_flow.py --model meta-llama/Llama-3.2-1B-Instruct
        --tag llama3.2-1b --st053 <dossier 0.5.3>
"""

import argparse
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

ARM_REF = "C3_opi"
ARM_ADV = "C4"
COUCHE = 4


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--seed", type=int, default=2027)
    ap.add_argument("--per-stratum", type=int, default=2)
    ap.add_argument("--max-tokens", type=int, default=512)
    ap.add_argument("--couche", type=int, default=COUCHE)
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
    from spectral_trust import GSPConfig

    from spectral_injection.opi import capture as cm
    from spectral_injection.opi import dataset as ds
    from spectral_injection.opi import regions6 as r6
    from spectral_injection.opi import templates as tm
    from spectral_injection.opi import wikifiller as wf
    from run_region_energy import MultiCapturer, build_arms, sample

    out_dir = os.path.join(args.out, args.tag, "demo")
    os.makedirs(out_dir, exist_ok=True)
    print(f"spectral-trust {st.__version__} <- {os.path.dirname(st.__file__)}")

    df, _ = ds.load()
    template, _b = tm.resolve(args.model)
    cap = MultiCapturer(args.model, dtype="float32",
                        max_tokens=args.max_tokens)

    def n_tokens(instruction, external):
        txt = cm.format_user_only(cap.tokenizer,
                                  cm.build_user_turn(instruction, external),
                                  template)
        return len(cap.tokenizer(txt, add_special_tokens=False)["input_ids"])

    fillers = {"C3_opi": ds.BenignFiller(df, seed=args.seed),
               "C3_wiki": wf.WikitextFiller(seed=args.seed)}

    # la premiere paire du plan stratifie qui s'apparie : aucun choix
    ids, labs, row_ok = None, None, None
    for _, row in sample(df, args.per_stratum, args.seed).iterrows():
        arms = build_arms(row, fillers, n_tokens)
        if arms is None:
            continue
        segs, labels, ok = {}, {}, True
        for arm in (ARM_REF, ARM_ADV):
            seg = cm.segment(cap.tokenizer, row.instruction,
                             arms["texts"][arm], template)
            lab = r6.label_tokens(seg, row.instruction, arms["comps"][arm])
            if not np.array_equal(r6.to_three(lab), seg.regions):
                ok = False
                break
            segs[arm], labels[arm] = seg, np.asarray(lab)
        if not ok or len({len(segs[a]) for a in segs}) != 1:
            continue
        if not np.array_equal(labels[ARM_REF], labels[ARM_ADV]):
            continue
        ids = {a: list(segs[a].input_ids) for a in segs}
        labs = labels[ARM_REF]
        row_ok = row
        break
    if ids is None:
        print("aucune paire appariee trouvee")
        return 1

    T = len(ids[ARM_REF])
    islot = np.flatnonzero(labs == "slot")
    print(f"paire {row_ok.attack_type}__{row_ok.task_type}  T={T}  "
          f"slot={islot.size} tokens (positions {islot.min()}..{islot.max()})")

    sm = st.SpectralModel.from_pretrained(
        cap.model, tokenizer=cap.tokenizer,
        config=GSPConfig(symmetrization="symmetric", normalization="sym",
                         remove_self_loops=False))
    # les deux bras dans un seul cache : le lot garde l'appariement
    cache = sm.run_with_spectra(
        torch.tensor([ids[ARM_REF], ids[ARM_ADV]]), hybrid=True,
        keep_patterns=True)
    print(f"cache : {cache.n_texts} textes, {len(cache.layers)} couches, "
          f"{len(cache.metric_names)} metriques")
    print("metriques :", ", ".join(sorted(cache.metric_names)))

    D = cache.to_dataframe()
    D.to_csv(os.path.join(out_dir, "metriques.csv"), index=False)
    print(f"metriques.csv  {D.shape}")

    cache.save(os.path.join(out_dir, "cache.npz"))
    faits = []
    # -- toutes les vues animees/interactives annoncees par animation.KINDS,
    #    chacune en .html pour rester sans dependance video
    from spectral_trust.animation import KINDS
    print(f"\nKINDS annonces : {KINDS}")
    for kind in KINDS:
        nom = f"{kind}.html"
        p = os.path.join(out_dir, nom)
        appar = ""
        try:
            cache.animate(kind, p, text=1, other=cache.text(0), theme="light")
        except TypeError:
            try:
                cache.animate(kind, p, text=1, theme="light")
                appar = ", sans appariement"
            except Exception as e:
                faits.append(f"{nom:<20} ECHEC {type(e).__name__}: "
                             f"{str(e)[:90]}")
                continue
        except Exception as e:
            faits.append(f"{nom:<20} ECHEC {type(e).__name__}: {str(e)[:90]}")
            continue
        faits.append(f"{nom:<20} {os.path.getsize(p) // 1024} Ko{appar}")
    # -- les vues inline (SVG), rendues pour les deux bras
    inline = [
        ("heatmap_fiedler", lambda t: cache.heatmap(metric="fiedler_value",
                                                    text=t)),
        ("heatmap_hfer", lambda t: cache.heatmap(metric="hfer", text=t)),
        ("spectre_couche", lambda t: cache.spectrum(layer=args.couche,
                                                    text=t)),
        ("graphe_couche", lambda t: cache.graph(layer=args.couche, head=0,
                                                text=t)),
    ]
    for base, fn in inline:
        for t, bras in ((0, ARM_REF), (1, ARM_ADV)):
            nom = f"{base}__{bras}.svg"
            p = os.path.join(out_dir, nom)
            try:
                view = fn(t)
                svg = getattr(view, "svg", None)
                if not svg:
                    r = getattr(view, "_repr_svg_", None)
                    svg = r() if r else None
                if svg:
                    open(p, "w", encoding="utf-8").write(svg)
                    faits.append(f"{nom:<34} {os.path.getsize(p) // 1024} Ko")
                else:
                    faits.append(f"{nom:<34} pas de SVG rendu")
            except Exception as e:
                faits.append(f"{nom:<34} ECHEC {type(e).__name__}: "
                             f"{str(e)[:85]}")
    print("\n".join("  " + f for f in faits))

    # ou tombe le partage de Fiedler par rapport au slot : une observation,
    # pas une conclusion (une seule paire)
    # Le cache etiquette lui-meme chaque metrique 'attention' ou 'hybrid' :
    # la separation des familles est native en 0.5.3.
    try:
        print("\nfamilles declarees par le cache :")
        print(D.groupby("family").metric.unique().to_string())
        fv = D[(D.text_id == 1) & (D.layer == args.couche)
               & (D.metric == "attn.fiedler_value")]
        print(f"\ncouche {args.couche}, bras injecte, attn.fiedler_value : "
              f"{len(fv)} tetes, median {fv.value.median():.4e}, "
              f"min {fv.value.min():.4e}, max {fv.value.max():.4e}")
    except Exception as e:
        print("lecture Fiedler:", type(e).__name__, e)
    print(f"\n-> {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
