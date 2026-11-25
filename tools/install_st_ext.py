"""Installe les extensions du depot DANS le paquet ``spectral_trust`` installe.

Pourquoi ce detour. ``spectral_trust`` n'existe sur cette machine que sous
forme installee (site-packages) : il n'y a pas de checkout source. Ecrire
directement dans site-packages rendrait le code invisible au depot et le
ferait disparaitre au premier ``pip install``. La source canonique vit donc
dans ``spectral_trust_ext/`` et ce script la copie dans le paquet.

Operations, toutes IDEMPOTENTES et REVERSIBLES :

1. copie ``spectral_trust_ext/<mod>.py`` vers ``spectral_trust/<mod>.py`` ;
2. ajoute un bloc d'export delimite par des marqueurs dans
   ``spectral_trust/__init__.py``, apres avoir sauvegarde l'original en
   ``__init__.py.orig`` s'il n'existe pas deja.

``--uninstall`` retire les modules copies et restaure le bloc d'export.
``--check`` ne modifie rien et rend un diagnostic.
"""

import argparse
import hashlib
import os
import shutil
import sys

MODULES = ["region_energy", "dirichlet"]
BEGIN = "# >>> spectral_injection extensions (generated, do not edit) >>>"
END = "# <<< spectral_injection extensions <<<"

EXPORT_BLOCK = """
{begin}
from .region_energy import (
    RegionEnergy, region_energy, region_energy_all_layers,
    normalized_adjacency, check_variance_identity,
    block_names, block_pairs, region_indices,
)

from .dirichlet import (
    DirichletEnergy, dirichlet_energy, dirichlet_energy_all_layers,
    per_head_laplacians, head_aggregation_gap,
)

__all__ = list(__all__) + [
    "RegionEnergy", "region_energy", "region_energy_all_layers",
    "normalized_adjacency", "check_variance_identity",
    "block_names", "block_pairs", "region_indices",
    "DirichletEnergy", "dirichlet_energy", "dirichlet_energy_all_layers",
    "per_head_laplacians", "head_aggregation_gap",
]
{end}
"""


def sha(path: str) -> str:
    with open(path, "rb") as fh:
        return hashlib.sha256(fh.read()).hexdigest()[:16]


def package_dir() -> str:
    import spectral_trust
    return os.path.dirname(spectral_trust.__file__)


def strip_block(text: str) -> str:
    if BEGIN not in text:
        return text
    head, rest = text.split(BEGIN, 1)
    if END not in rest:
        return head.rstrip() + "\n"
    _, tail = rest.split(END, 1)
    return (head.rstrip() + "\n" + tail.lstrip("\n")).rstrip() + "\n"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--uninstall", action="store_true")
    ap.add_argument("--check", action="store_true")
    args = ap.parse_args()

    here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    src_dir = os.path.join(here, "spectral_trust_ext")
    pkg = package_dir()
    init = os.path.join(pkg, "__init__.py")
    backup = init + ".orig"

    if args.check:
        print(f"paquet   : {pkg}")
        for m in MODULES:
            s = os.path.join(src_dir, m + ".py")
            d = os.path.join(pkg, m + ".py")
            ok = os.path.exists(d) and sha(s) == sha(d)
            print(f"  {m:18s} depot {sha(s)}  paquet "
                  f"{sha(d) if os.path.exists(d) else '-- absent --':16s}  "
                  f"{'a jour' if ok else 'DIVERGENT'}")
        with open(init, encoding="utf-8") as fh:
            print(f"  export dans __init__ : "
                  f"{'present' if BEGIN in fh.read() else 'absent'}")
        print(f"  sauvegarde __init__  : "
              f"{'presente' if os.path.exists(backup) else 'absente'}")
        return 0

    if args.uninstall:
        for m in MODULES:
            d = os.path.join(pkg, m + ".py")
            if os.path.exists(d):
                os.remove(d)
                print(f"retire {d}")
        with open(init, encoding="utf-8") as fh:
            txt = fh.read()
        if BEGIN in txt:
            with open(init, "w", encoding="utf-8") as fh:
                fh.write(strip_block(txt))
            print("bloc d'export retire")
        return 0

    if not os.path.exists(backup):
        shutil.copy2(init, backup)
        print(f"sauvegarde -> {backup}")

    for m in MODULES:
        s = os.path.join(src_dir, m + ".py")
        d = os.path.join(pkg, m + ".py")
        if not os.path.exists(s):
            raise SystemExit(f"source absente : {s}")
        shutil.copy2(s, d)
        print(f"copie {m}.py  sha {sha(d)}")

    with open(init, encoding="utf-8") as fh:
        txt = fh.read()
    txt = strip_block(txt)
    txt = txt.rstrip() + "\n" + EXPORT_BLOCK.format(begin=BEGIN, end=END)
    with open(init, "w", encoding="utf-8") as fh:
        fh.write(txt)
    print("bloc d'export (re)ecrit")

    # verification a chaud, dans un interpreteur neuf
    import subprocess
    r = subprocess.run(
        [sys.executable, "-c",
         "import spectral_trust as st; "
         "print('import OK', st.__version__, "
         "hasattr(st, 'region_energy_all_layers'))"],
        capture_output=True, text=True)
    print(r.stdout.strip() or r.stderr.strip()[-400:])
    return 0 if r.returncode == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
