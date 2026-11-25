"""Correctif de precision pour ``GraphConstructor.construct_laplacian``.

ETAT INITIAL. La methode contient

    # Ensure non-negative weights and promote to Float32 to avoid BFloat16-CPU issues
    adjacency = torch.clamp(adjacency, min=0).to(torch.float32)

Le commentaire dit << promote >>, mais ``.to(torch.float32)`` est
inconditionnel : une entree float64 est **degradee**. Mesure sur une
attention reelle (Qwen2.5-1.5B, couche 14, 12 tetes, N = 104) :

    ecart M bibliotheque vs float64 pur    2.14e-07
    ecart float32 vs float64 pur           2.92e-08

L'energie regionale est une somme de N^2 carres ; a N = 512 le plancher de
precision atteint ~1e-8, du meme ordre que les effets mesures (Delta de
l'ordre de 1e-4 a 1e-3). La marge reste large, mais elle est inutilement
reduite, et l'equivalence avec un calcul de reference float64 ne peut plus
etre etablie mieux qu'a 1e-8.

CORRECTIF. Respecter l'intention du commentaire : promouvoir les dtypes de
precision inferieure a float32, laisser float64 intact.

    adjacency = torch.clamp(adjacency, min=0)
    if adjacency.dtype not in (torch.float32, torch.float64):
        adjacency = adjacency.to(torch.float32)

Le comportement est INCHANGE pour toute entree float32, float16 ou
bfloat16, c'est-a-dire pour tout appel existant de la bibliotheque, dont
``per_head_metrics`` qui passe du float32. Seules les entrees float64 --
qui n'existaient pas avant ce correctif, puisqu'elles etaient degradees --
voient une difference.

Le correctif est applique par substitution de chaine exacte, refuse de
s'appliquer deux fois, et ``--revert`` restaure la ligne d'origine.
"""

import argparse
import os
import shutil

OLD = ("        # Ensure non-negative weights and promote to Float32 to "
       "avoid BFloat16-CPU issues\n"
       "        adjacency = torch.clamp(adjacency, min=0).to(torch.float32)\n")

NEW = ("        # Ensure non-negative weights and promote LOW-precision dtypes to\n"
       "        # Float32 to avoid BFloat16-CPU issues. Patched by\n"
       "        # spectral_trust_ext/patch_graph_dtype.py: float64 input is no\n"
       "        # longer downcast, which capped region-energy precision at ~1e-8.\n"
       "        adjacency = torch.clamp(adjacency, min=0)\n"
       "        if adjacency.dtype not in (torch.float32, torch.float64):\n"
       "            adjacency = adjacency.to(torch.float32)\n")

MARK = "spectral_trust_ext/patch_graph_dtype.py"


def graph_path() -> str:
    import spectral_trust
    return os.path.join(os.path.dirname(spectral_trust.__file__), "graph.py")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--revert", action="store_true")
    ap.add_argument("--check", action="store_true")
    args = ap.parse_args()

    p = graph_path()
    with open(p, encoding="utf-8") as fh:
        txt = fh.read()
    applied = MARK in txt

    if args.check:
        print(f"{p}\n  correctif : {'APPLIQUE' if applied else 'absent'}")
        print(f"  ligne d'origine presente : {OLD in txt}")
        return 0

    if args.revert:
        if not applied:
            print("correctif absent, rien a faire")
            return 0
        if NEW not in txt:
            raise SystemExit("le bloc patche a ete modifie a la main ; "
                             "restaurer depuis graph.py.orig")
        with open(p, "w", encoding="utf-8") as fh:
            fh.write(txt.replace(NEW, OLD))
        print("correctif retire")
        return 0

    if applied:
        print("correctif deja applique, rien a faire (idempotent)")
        return 0
    if OLD not in txt:
        raise SystemExit(
            "la ligne attendue est absente de graph.py : la version de "
            "spectral-trust a change, verifier manuellement avant de patcher")
    if not os.path.exists(p + ".orig"):
        shutil.copy2(p, p + ".orig")
        print(f"sauvegarde -> {p}.orig")
    with open(p, "w", encoding="utf-8") as fh:
        fh.write(txt.replace(OLD, NEW, 1))
    print("correctif applique")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
