"""Figures de la campagne d'energie regionale.

ECHELLES DE COULEUR. Deux cartes mises cote a cote partagent toujours leurs
bornes. Entre MODELES, l'echelle est commune par quantite : sans cela une
carte pale et une carte vive se compareraient a l'oeil alors que leurs
amplitudes diffèrent d'un ordre de grandeur. Les bornes sont symetriques
autour de zero (``RdBu_r``), parce que le signe est la lecture principale.

AUCUNE MOYENNE DE TETES. Chaque pixel est une cellule (couche, tete) ;
la valeur affichee est la MEDIANE sur les paires du Delta de cette cellule,
jamais une moyenne sur les tetes.

Les modeles n'ont ni le meme nombre de couches ni le meme nombre de tetes.
Les cartes ne sont donc PAS redimensionnees a une grille commune -- cela
inventerait des cellules. Chaque carte garde ses dimensions propres, et
l'axe vertical est annote en profondeur relative pour que la lecture reste
comparable.
"""

import argparse
import json
import os
import sys
from typing import Dict, List

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def _maps_from_frame(f: pd.DataFrame, group_cols: List[str], val: str,
                     arm: str) -> Dict[str, np.ndarray]:
    """``[paires, bras, couche, tete]`` -> {quantite: [couches, tetes]}.

    Un SEUL passage pour toutes les quantites du fichier. La version
    precedente rechargeait le parquet une fois par quantite ET par modele,
    soit jusqu'a 40 lectures de 26 Mo pour une figure.
    """
    f = f[f.arm.isin(["C4", arm])]
    piv = f.pivot_table(index=group_cols + ["couche", "tete"], columns="arm",
                        values=val)
    if "C4" not in piv.columns or arm not in piv.columns:
        return {}
    delta = (piv["C4"] - piv[arm]).rename("_d").reset_index()
    out = {}
    for key, g in delta.groupby(group_cols):
        name = "|".join(map(str, key)) if isinstance(key, tuple) else str(key)
        m = g.groupby(["couche", "tete"])["_d"].median().unstack("tete")
        out[name] = m.to_numpy()
    return out


def model_maps(d: str, arm: str) -> Dict[str, Dict[str, np.ndarray]]:
    """Toutes les cartes d'un modele, en deux lectures de parquet."""
    out = {}
    bl = pd.read_parquet(os.path.join(d, "blocks_long.parquet"))
    out["block_energy"] = _maps_from_frame(bl, ["bloc"], "energy", arm)
    out["block_share"] = _maps_from_frame(bl, ["bloc"], "share", arm)
    del bl
    fl = pd.read_parquet(os.path.join(d, "flows_long.parquet"))
    out["directed_flow"] = _maps_from_frame(fl, ["query", "key"], "flow", arm)
    del fl
    return out


def grid(maps: Dict[str, np.ndarray], title: str, out: str, sub: str = ""):
    """Une rangee de cartes, bornes de couleur COMMUNES."""
    names = list(maps)
    v = max((float(np.nanmax(np.abs(m))) for m in maps.values()), default=1e-9)
    v = v or 1e-9
    widths = [max(2.2, maps[n].shape[1] * 0.16) for n in names]
    fig, axes = plt.subplots(1, len(names), figsize=(sum(widths) + 2.2, 4.6),
                             gridspec_kw={"width_ratios": widths})
    if len(names) == 1:
        axes = [axes]
    for ax, n in zip(axes, names):
        M = maps[n]
        im = ax.imshow(M, cmap="RdBu_r", vmin=-v, vmax=v, aspect="auto",
                       interpolation="nearest", origin="lower")
        L, H = M.shape
        ax.set_title(f"{n}\n{L} x {H}", fontsize=9)
        ax.set_xlabel("tête", fontsize=8)
        ax.set_xticks([0, H - 1])
        ax.set_xticklabels(["0", str(H - 1)], fontsize=7)
        ax.set_yticks([0, L - 1])
        ax.set_yticklabels(["0\n(entrée)", f"{L - 1}\n(sortie)"], fontsize=7)
    axes[0].set_ylabel("couche", fontsize=8)
    cb = fig.colorbar(im, ax=axes, fraction=0.025, pad=0.02)
    cb.set_label("Δ médian par cellule (C4 − C3), échelle commune", fontsize=8)
    cb.ax.tick_params(labelsize=7)
    fig.suptitle(title + ("\n" + sub if sub else ""), fontsize=11)
    fig.savefig(out, dpi=140, bbox_inches="tight")
    plt.close(fig)
    return v


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=os.path.join("results", "opi",
                                                   "region_energy"))
    ap.add_argument("--arm", default="C3_opi")
    ap.add_argument("--top", type=int, default=4)
    args = ap.parse_args()
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

    fig_dir = os.path.join(args.root, "figures")
    os.makedirs(fig_dir, exist_ok=True)
    rep = pd.read_csv(os.path.join(args.root, "reproducibility.csv"))
    tags = [t for t in sorted(os.listdir(args.root))
            if os.path.exists(os.path.join(args.root, t, "provenance.json"))]

    # -- quantites a tracer : les reproduites d'abord, sinon les plus fortes
    chosen: List[tuple] = []
    for fam in ("block_energy", "block_share", "directed_flow"):
        sub = rep[rep.family == fam]
        if sub.empty:
            continue
        pick = sub[sub.unanimous]
        if pick.empty:
            pick = sub.sort_values("n_robust", ascending=False)
        pick = pick.reindex(pick.median_delta_opi.abs()
                            .sort_values(ascending=False).index)
        for _, r in pick.head(args.top).iterrows():
            chosen.append((fam, r.quantity, bool(r.unanimous)))

    # -- un seul chargement de parquet par modele, pour toutes les figures
    cache: Dict[str, Dict[str, Dict[str, np.ndarray]]] = {}
    for t in tags:
        cache[t] = model_maps(os.path.join(args.root, t), args.arm)
        print(f"  cartes chargees : {t}")

    made = []
    for fam, q, una in chosen:
        maps = {t: cache[t][fam][q] for t in tags
                if q in cache[t].get(fam, {})
                and cache[t][fam][q].size > 1}
        if not maps:
            continue
        safe = q.replace("|", "_to_").replace("--", "_")
        out = os.path.join(fig_dir, f"{fam}__{safe}.png")
        label = q.replace("|", " → ")
        v = grid(maps, f"{label}   ({fam})", out,
                 sub=("reproduite dans tous les modèles"
                      if una else "non reproduite partout")
                 + f"   —   bras {args.arm}")
        made.append((out, label, v))
        print(f"  {out}   borne ±{v:.3g}")

    print(f"\n{len(made)} figures dans {fig_dir}")
    with open(os.path.join(fig_dir, "index.json"), "w", encoding="utf-8") as fh:
        json.dump([{"file": os.path.basename(o), "quantity": l,
                    "color_bound": v} for o, l, v in made], fh, indent=2)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
