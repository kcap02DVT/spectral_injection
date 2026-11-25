"""Figures. Echelles de couleur COMMUNES quand deux matrices sont comparees ;
distributions sur TOUTES les tetes ; vecteur de Fiedler sur UNE tete declaree."""

from typing import Optional, Sequence

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def adjacency_pair(W_adv: np.ndarray, W_ctl: np.ndarray, reg_adv: np.ndarray,
                   reg_ctl: np.ndarray, title: str, out: str):
    """Deux matrices W, UNE echelle de couleur (vmin/vmax partages)."""
    vmax = float(max(W_adv.max(), W_ctl.max()))
    fig, axes = plt.subplots(1, 2, figsize=(11, 5))
    for ax, W, reg, lab in zip(axes, (W_adv, W_ctl), (reg_adv, reg_ctl),
                               ("adverse", "controle")):
        im = ax.imshow(W, cmap="magma", vmin=0.0, vmax=vmax, interpolation="nearest")
        for r, col in (("S", "cyan"), ("U", "lime")):
            idx = np.flatnonzero(reg == r)
            if idx.size:
                ax.axhline(idx.min() - .5, color=col, lw=.8)
                ax.axhline(idx.max() + .5, color=col, lw=.8)
                ax.axvline(idx.min() - .5, color=col, lw=.8)
                ax.axvline(idx.max() + .5, color=col, lw=.8)
        ax.set_title(f"{lab}  (T={W.shape[0]})")
        ax.set_xticks([]); ax.set_yticks([])
    fig.colorbar(im, ax=axes, fraction=.03, label="W = (A + Aᵀ)/2, échelle commune")
    fig.suptitle(title); fig.savefig(out, dpi=130, bbox_inches="tight"); plt.close(fig)


def lambda2_distribution(long_df: pd.DataFrame, metric: str, title: str, out: str):
    """Toutes les cellules (paire x couche x tete), par condition. Chaque
    point est une tete : rien n'est moyenne."""
    sub = long_df[long_df.metrique == metric].dropna(subset=["valeur"])
    fig, ax = plt.subplots(figsize=(7, 4.5))
    conds = ["CTL", "ADV"]
    data = [sub[sub.condition == c].valeur.to_numpy() for c in conds]
    ax.violinplot(data, positions=[0, 1], showmedians=True, widths=.8)
    rng = np.random.default_rng(0)
    for i, d in enumerate(data):
        ax.scatter(i + rng.uniform(-.28, .28, d.size), d, s=3, alpha=.15, color="k")
    ax.set_xticks([0, 1]); ax.set_xticklabels([f"contrôle\n(n={data[0].size})",
                                               f"adverse\n(n={data[1].size})"])
    ax.set_ylabel(metric); ax.set_title(title + "\n(une observation = une tête ; aucune moyenne)")
    fig.savefig(out, dpi=130, bbox_inches="tight"); plt.close(fig)


def delta_map(cells: pd.DataFrame, n_layers: int, n_heads: int, title: str, out: str,
              mark: str = "sig_bh"):
    """Carte couche x tete du delta MEDIAN apparie ; cellules significatives
    (BH) marquees d'un point, FWER d'une croix."""
    M = np.full((n_layers, n_heads), np.nan)
    for _, r in cells.iterrows():
        M[r.couche, r.tete] = r.delta_median
    v = float(np.nanmax(np.abs(M))) or 1e-9
    fig, ax = plt.subplots(figsize=(max(6, n_heads * .45), max(5, n_layers * .32)))
    im = ax.imshow(M, cmap="RdBu_r", vmin=-v, vmax=v, aspect="auto")
    for _, r in cells.iterrows():
        if r.sig_fwer:
            ax.plot(r.tete, r.couche, "kx", ms=6, mew=1.5)
        elif r.get(mark, False):
            ax.plot(r.tete, r.couche, "k.", ms=4)
    ax.set_xlabel("tête"); ax.set_ylabel("couche")
    ax.set_xticks(range(n_heads)); ax.set_yticks(range(n_layers))
    fig.colorbar(im, ax=ax, label="Δ médian (adverse − contrôle)")
    ax.set_title(f"{title}\n· = BH q<.05   × = max-stat FWER<.05")
    fig.savefig(out, dpi=130, bbox_inches="tight"); plt.close(fig)


def fiedler_plot(u_adv: np.ndarray, u_ctl: np.ndarray, reg_adv: np.ndarray,
                 reg_ctl: np.ndarray, lam_adv: float, lam_ctl: float,
                 cell: tuple, title: str, out: str):
    """u_2 le long des tokens pour la cellule DECLAREE, adverse et controle
    superposes, regions ombrees. Jamais une moyenne de tetes."""
    fig, axes = plt.subplots(2, 1, figsize=(11, 5.5), sharey=True)
    for ax, u, reg, lab, lam in zip(axes, (u_adv, u_ctl), (reg_adv, reg_ctl),
                                    ("adverse", "contrôle"), (lam_adv, lam_ctl)):
        x = np.arange(u.size)
        # Ombrage par PLAGES CONTIGUES : T (template) n'est pas contigu --
        # prefixe en tete, suffixe en queue -- et ombrer de min a max
        # recouvrirait S et U. Defaut constate sur la premiere execution.
        for r, col in (("T", "#eeeeee"), ("S", "#cfefff"), ("U", "#d9f7d9")):
            idx = np.flatnonzero(reg == r)
            if not idx.size:
                continue
            cuts = np.flatnonzero(np.diff(idx) > 1)
            for run in np.split(idx, cuts + 1):
                ax.axvspan(run.min() - .5, run.max() + .5, color=col, zorder=0)
        ax.plot(x, u, lw=1.2, color="k"); ax.axhline(0, color="gray", lw=.6)
        ax.set_ylabel("u₂"); ax.set_title(f"{lab} — λ₂ = {lam:.4f}", loc="left", fontsize=10)
    axes[-1].set_xlabel("position du token  (bleu = S, vert = U, gris = template)")
    fig.suptitle(f"{title} — vecteur de Fiedler, couche {cell[0]} tête {cell[1]} (déclarée)")
    fig.savefig(out, dpi=130, bbox_inches="tight"); plt.close(fig)
