"""Lecture aveugle d'un run de test : seuls les triplets pré-enregistrés.

    python -m spectral_injection.verdict results/bipia --split test

Le CLI imprime le classement complet des triplets, qui est précisément ce
qu'il ne faut pas regarder sur le split d'évaluation : y prendre le maximum
réintroduit le biais de sélection que le protocole train/test existe pour
éliminer. Ce module ne lit que la ligne fixée d'avance, et ignore le reste.

Les triplets ci-dessous ont été arrêtés sur le split ``train``, avant tout
calcul sur ``test``. Ils ne doivent plus être modifiés.
"""

import argparse
import csv
import os
import sys
from typing import Dict, Optional

import numpy as np

#: (couche, tête, métrique) fixés sur train, avec l'AUROC obtenue alors.
PREREGISTRE: Dict[str, Dict] = {
    # Relancé le 15/09 : 0.8856 contre 0.8838 au premier run. Le triplet est
    # inchangé ; l'écart vient du non-déterminisme float32 du calcul
    # d'attention, soit environ trois inversions de rang sur 1600 paires.
    "Llama-3.2-1B-Instruct": dict(layer=6,  head=21, metric="connectivity_ratio", train=0.8856),
    "Llama-3.2-3B-Instruct": dict(layer=14, head=4,  metric="fiedler_value",      train=0.9169),
    "Qwen2.5-1.5B-Instruct": dict(layer=9,  head=9,  metric="connectivity_ratio", train=0.8131),
    "Qwen2.5-3B-Instruct":   dict(layer=18, head=10, metric="fiedler_value",      train=0.8756),
    "Qwen3-1.7B":            dict(layer=17, head=6,  metric="fiedler_value",      train=0.9069),
    "SmolLM2-1.7B-Instruct": dict(layer=17, head=3,  metric="fiedler_value",      train=0.8575),
    "SmolLM3-3B":            dict(layer=24, head=0,  metric="spectral_radius",    train=0.7913),
    "gemma-2-2b-it":         dict(layer=9,  head=0,  metric="connectivity_ratio", train=0.7794),
}


def lire(path: str, spec: Dict) -> Optional[Dict]:
    """La seule ligne qui nous intéresse, sans jamais trier le fichier."""
    with open(path, newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            if (int(row["layer"]) == spec["layer"]
                    and int(row["head"]) == spec["head"]
                    and row["metric"] == spec["metric"]):
                return row
    return None


def main(argv=None) -> int:
    p = argparse.ArgumentParser(
        description="Verdict du split de test sur les triplets pré-enregistrés.")
    p.add_argument("base", nargs="?", default="results/bipia")
    p.add_argument("--split", default="test")
    args = p.parse_args(argv)

    print(f"{'modèle':<24}{'triplet fixé sur train':<32}"
          f"{'train':>8}{'test':>8}{'écart':>8}")
    print("-" * 80)

    tests = []
    for nom, spec in sorted(PREREGISTRE.items()):
        path = os.path.join(args.base, nom, args.split, "per_head_auroc.csv")
        tag = f"L{spec['layer']} H{spec['head']} {spec['metric'][:16]}"
        if not os.path.exists(path):
            print(f"{nom:<24}{tag:<32}{spec['train']:>8.3f}{'--':>8}"
                  f"{'run absent':>8}")
            continue
        row = lire(path, spec)
        if row is None:
            print(f"{nom:<24}{tag:<32}{spec['train']:>8.3f}{'--':>8}"
                  f"{'introuvable':>8}")
            continue
        a = float(row["auroc"])
        tests.append(a)
        print(f"{nom:<24}{tag:<32}{spec['train']:>8.3f}{a:>8.3f}"
              f"{a - spec['train']:>+8.3f}")

    if not tests:
        print("\nAucun run de test exploitable.", file=sys.stderr)
        return 1

    a = np.array(tests)
    print("-" * 80)
    print(f"{'médiane':<24}{'':<32}{'':>8}{np.median(a):>8.3f}")
    print(f"\n{len(a)} modèle(s) testé(s) — "
          f"{int((a >= 0.75).sum())} au-dessus de 0,75, "
          f"{int((a >= 0.70).sum())} au-dessus de 0,70, "
          f"{int((a < 0.60).sum())} sous 0,60.")
    print("\nUn seul candidat par modèle, fixé avant ce run : aucune correction")
    print("pour sélection n'est requise sur ces chiffres.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
