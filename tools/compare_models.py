"""Comparaison inter-modeles du contraste principal C4 - C3. Lecture seule.

    python tools/compare_models.py

Chaque modele est traite comme sa propre famille : la correction
max-statistic porte sur les 12 metriques confirmatoires **a l'interieur**
d'un modele, jamais a travers les modeles. Corriger a travers reviendrait a
poser une autre question -- << l'effet existe-t-il quelque part >> -- alors
que la question posee est << l'effet tient-il pour ce modele-ci >>.

Les 112 exemples sont les memes pour tous les modeles (meme parquet, meme
graine, empreinte d'index identique), donc la comparaison est appariee au
niveau de l'exemple. Ce qui change d'un modele a l'autre :

    la famille de tokenizer, donc la segmentation en tokens ;
    le template experimental, donc la taille de la region T ;
    la profondeur et le nombre de tetes, donc le nombre de cellules agregees.

Aucune de ces differences n'est controlee. Une divergence entre modeles ne
dit donc pas ou elle prend sa source.
"""

import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, ".")
from spectral_injection.opi.stats import (CONFIRMATORY, CONFIRMATORY_TRAJ,
                                          aggregate_examples,
                                          aggregate_trajectories,
                                          paired_differences, sign_flip_test)

#: (etiquette, repertoire). Le pilote d'hier sert de reference.
MODELS = [
    ("llama3.2-1b", "results/opi/pilot"),
    ("qwen2.5-1.5b", "results/opi/multi/qwen2.5-1.5b"),
    ("gemma-2-2b", "results/opi/multi/gemma-2-2b"),
    ("llama3.2-3b", "results/opi/multi/llama3.2-3b"),
]
N_PERM = 20000
SEED = 0


def load(path):
    d = np.load(os.path.join(path, "values.npz"), allow_pickle=True)
    names = [str(x) for x in d["metrics"]] + [str(x) for x in d["traj_metrics"]]
    conds = [str(x) for x in d["conditions"]]
    A = np.concatenate([aggregate_examples(d["values"]),
                        aggregate_trajectories(d["trajectories"])], axis=2)
    return A, names, conds, [int(x) for x in d["row_index"]]


def main() -> int:
    fam = list(CONFIRMATORY) + list(CONFIRMATORY_TRAJ)
    res, diffs, shapes, ref_rows = {}, {}, {}, None

    for label, path in MODELS:
        if not os.path.exists(os.path.join(path, "values.npz")):
            print(f"[absent] {label} : {path}")
            continue
        A, names, conds, rows = load(path)
        if ref_rows is None:
            ref_rows = rows
        elif rows != ref_rows:
            raise RuntimeError(
                f"{label} : les exemples different de la reference ; "
                f"la comparaison appariee serait fausse")
        miss = [f for f in fam if f not in names]
        if miss:
            raise RuntimeError(f"{label} : metriques absentes {miss}")
        idx = [names.index(f) for f in fam]
        pdv = paired_differences(A[:, :, idx], fam, conds, "C4", "C3")
        res[label] = sign_flip_test(pdv, n_perm=N_PERM, seed=SEED)
        diffs[label] = pdv.diff
        shapes[label] = A.shape
        print(f"[charge] {label:<14} n={pdv.diff.shape[0]:>4}  "
              f"metriques={A.shape[2]}")

    if len(res) < 2:
        print("\nmoins de deux modeles disponibles : rien a comparer")
        return 0

    labels = list(res)
    print("\n" + "=" * 112)
    print("MEDIANE DE C4 - C3   (win_rate entre parentheses)")
    hdr = f"{'metrique':<34}" + "".join(f"{l:>19}" for l in labels)
    print(hdr); print("-" * len(hdr))
    for j, f in enumerate(fam):
        row = f"{f:<34}"
        for l in labels:
            D = diffs[l][:, j]
            row += f"{np.median(D):>12.4f}({(D > 0).mean():>4.2f})"
        print(row)

    print("\n" + "=" * 112)
    print("p CORRIGEE (max-statistic, famille de 12, INTERNE a chaque modele)")
    print(hdr); print("-" * len(hdr))
    for j, f in enumerate(fam):
        row = f"{f:<34}"
        for l in labels:
            p = res[l][j]["p_fwer"]
            row += f"{p:>17.4f}{'*' if p < 0.05 else ' '} "
        print(row)

    print("\n" + "=" * 112)
    print("CONCORDANCE DE DIRECTION ENTRE MODELES")
    print("Une mediane exactement nulle n'a pas de direction : elle est")
    print("comptee comme 'nulle partout' (accord) et non comme un desaccord.")
    print()
    print(f"{'metrique':<34}{'direction':>20}{'significatif partout':>24}")
    print("-" * 80)
    n_coh = n_zero = 0
    for j, f in enumerate(fam):
        sg = [np.sign(np.median(diffs[l][:, j])) for l in labels]
        sig = sum(res[l][j]["p_fwer"] < 0.05 for l in labels)
        n = len(labels)
        # ``sign(0)`` ne vaut ni +1 ni -1. Compter seulement les signes non
        # nuls ferait passer une metrique degenerate dans TOUS les modeles
        # pour un desaccord, alors que les modeles sont d'accord : la
        # quantite ne bouge pas. Defaut constate sur h5_k3_mix_UE.
        if all(s == 0.0 for s in sg):
            label, same, n_zero = f"nulle partout ({n}/{n})", n, n_zero + 1
        else:
            same = max(sg.count(1.0), sg.count(-1.0))
            label = f"{same}/{n} meme signe"
            if same == n and sig == n:
                n_coh += 1
        print(f"{f:<34}{label:>20}{sig:>21}/{n}")
    print(f"\n{n_coh}/{len(fam)} metriques de meme signe ET significatives "
          f"sur les {len(labels)} modeles")
    print(f"{n_zero}/{len(fam)} metriques de mediane nulle dans tous les "
          f"modeles (degenerescence structurelle, pas un desaccord)")
    agree = n_coh + n_zero + sum(
        1 for j in range(len(fam))
        if not all(np.sign(np.median(diffs[l][:, j])) == 0.0 for l in labels)
        and max([np.sign(np.median(diffs[l][:, j])) for l in labels].count(1.0),
                [np.sign(np.median(diffs[l][:, j])) for l in labels].count(-1.0))
        == len(labels)
        and sum(res[l][j]["p_fwer"] < 0.05 for l in labels) != len(labels))
    print(f"{agree}/{len(fam)} metriques en accord de direction "
          f"(signe commun ou nullite commune), significativite mise a part")

    out = []
    for l in labels:
        for j, f in enumerate(fam):
            D = diffs[l][:, j]
            out.append({"model": l, "metric": f,
                        "median": float(np.median(D)),
                        "win_rate": float((D > 0).mean()),
                        "dz": res[l][j].get("dz"),
                        "p_fwer": res[l][j]["p_fwer"],
                        "n": int(D.shape[0])})
    path = "results/opi/multi/comparison.csv"
    pd.DataFrame(out).to_csv(path, index=False)
    print(f"\n-> {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
