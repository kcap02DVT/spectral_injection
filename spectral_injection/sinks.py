"""Tache 1 : quelle part du spectre les puits d'attention possedent-ils ?

Le depot porte deux topologies incompatibles sans l'avoir decide :

    G2  attention brute, puits et template inclus -- c'est la que lambda_2
        par tete atteint 0,83 hors distribution ;
    G1  puits et template retires -- c'est la que cut_conductance echoue.

Les deux ne peuvent pas etre discutees comme "le graphe d'attention". Ce
module mesure exactement ce que le retrait enleve, et ou il deplace le gap.

Trois facons de retirer un noeud, parce qu'elles ne sont pas equivalentes :

    drop     on supprime lignes et colonnes. La masse que le puits portait
             disparait : les noeuds gardes perdent du degre et le graphe
             restant est globalement plus pauvre. C'est ce que fait G1.
    renorm   on supprime, puis on remet chaque noeud garde a son degre
             d'origine en redistribuant proportionnellement sur ses aretes
             survivantes. La masse est conservee, la structure locale aussi.
    kron     reduction de Kron : complement de Schur du laplacien
             combinatoire sur le bloc garde,

                 L_red = L_KK - L_KS (L_SS)^-1 L_SK.

             C'est la seule des trois qui soit une operation de theorie des
             graphes et non un decoupage de matrice : L_red est encore un
             laplacien, ses poids restent positifs, et les resistances
             effectives entre noeuds gardes sont exactement preservees. Le
             chemin i -> puits -> j devient une arete i-j de conductance
             equivalente au lieu d'etre perdu. C'est la definition
             mathematique de "masse redistribuee".

Ce qui est quantifie, sur le laplacien symetrique normalise :

    part_k = somme des u_k(i)^2 sur les puits -- la part du vecteur propre k
             qui vit sur les hubs. Les u_k etant de norme 1, part_k est
             directement une fraction d'energie, dans [0, 1]. part_2 dit si
             le vecteur de Fiedler est un objet du graphe ou un objet du
             puits.
    vol_share = vol(S)/vol(V) : pour le laplacien combinatoire, l'energie de
             Dirichlet portee par les coordonnees des puits vaut exactement
             cette fraction, tr(P_S L P_S)/tr(L). Sur L_sym la meme trace
             degenere en |S|/T, d'ou le passage par le combinatoire.
    ipr    = somme u^4 / (somme u^2)^2 : son inverse donne le nombre effectif
             de noeuds sur lesquels u_2 s'etale. Un u_2 localise sur trois
             noeuds n'est pas une coupe, c'est un mode de hub.
    gap_index = argmax des ecarts lambda_{k+1} - lambda_k sur la partie basse
             du spectre : ou le spectre dit que le decoupage naturel se
             trouve. C'est cette localisation que le retrait deplace.

Entree : le cache d'attention, donc aucune passe de modele.

    python -m spectral_injection.sinks cache/attn/MODELE/SPLIT \\
        --layer-stride 4 --reduction kron
"""

import argparse
import os
import sys
from typing import Dict, List, Optional, Tuple

import numpy as np

from .prompts import ROLE_INJECTION, ROLE_TEMPLATE

EPS = 1e-12
LOW_BAND = 16          # combien de valeurs propres basses comptent comme basses


# --------------------------------------------------------------------------
# Construction
# --------------------------------------------------------------------------
def symmetrize(attn_layer: np.ndarray, per_head: bool = False) -> np.ndarray:
    """[H,T,T] -> W symetrique a diagonale nulle : [T,T] ou [H,T,T]."""
    A = np.asarray(attn_layer, dtype=np.float64)
    if A.ndim != 3:
        raise ValueError(f"attendu [H,T,T], recu {A.shape}")
    if not per_head:
        A = A.mean(0)[None]
    W = 0.5 * (A + np.transpose(A, (0, 2, 1)))
    idx = np.arange(W.shape[-1])
    W[:, idx, idx] = 0.0
    return W if per_head else W[0]


def sink_indices(W: np.ndarray, share: float = 0.02,
                 always_bos: bool = True) -> np.ndarray:
    """Indices des puits : BOS, l'argmax de masse recue, et tout noeud
    captant plus que ``share`` de la masse totale.

    Le seuil est relatif a la masse totale, pas au degre moyen : un puits
    d'attention ne depasse pas un peu la moyenne, il en capte l'ordre de
    20 %. Avec share=0.02 et T=245 le seuil est a cinq fois le degre moyen,
    ce qui ne ramasse rien par accident.
    """
    mass = W.sum(0)
    total = max(mass.sum(), EPS)
    idx = set(np.flatnonzero(mass / total >= share).tolist())
    idx.add(int(mass.argmax()))
    if always_bos:
        idx.add(0)
    return np.array(sorted(idx), dtype=int)


# --------------------------------------------------------------------------
# Les trois retraits
# --------------------------------------------------------------------------
def remove_drop(W: np.ndarray, sinks: np.ndarray) -> np.ndarray:
    """Suppression seche : exactement ce que fait ``graphs.keep_mask``."""
    keep = np.setdiff1d(np.arange(W.shape[0]), sinks)
    return W[np.ix_(keep, keep)]


def remove_renorm(W: np.ndarray, sinks: np.ndarray) -> np.ndarray:
    """Suppression, puis chaque noeud garde retrouve son degre d'origine.

    Le facteur est applique symetriquement (racine du produit des deux
    facteurs de ligne) pour que la matrice reste symetrique : une
    renormalisation par ligne seule produirait une matrice non symetrique
    dont le laplacien n'a plus de sens spectral.
    """
    keep = np.setdiff1d(np.arange(W.shape[0]), sinks)
    deg_before = W.sum(1)[keep]
    sub = W[np.ix_(keep, keep)]
    deg_after = sub.sum(1)
    f = np.where(deg_after > EPS, deg_before / np.maximum(deg_after, EPS), 1.0)
    return sub * np.sqrt(np.outer(f, f))


def remove_kron(W: np.ndarray, sinks: np.ndarray) -> np.ndarray:
    """Reduction de Kron : complement de Schur sur le laplacien combinatoire.

    L = D - W, puis L/L_SS = L_KK - L_KS L_SS^-1 L_SK. Le resultat est un
    laplacien ; on en relit les poids par W_red = -offdiag(L_red), les
    residus negatifs numeriques etant ramenes a zero.
    """
    n = W.shape[0]
    keep = np.setdiff1d(np.arange(n), sinks)
    L = np.diag(W.sum(1)) - W
    L_kk = L[np.ix_(keep, keep)]
    L_ks = L[np.ix_(keep, sinks)]
    L_ss = L[np.ix_(sinks, sinks)]
    red = L_kk - L_ks @ np.linalg.pinv(L_ss) @ L_ks.T
    red = 0.5 * (red + red.T)
    W_red = -red
    np.fill_diagonal(W_red, 0.0)
    return np.clip(W_red, 0.0, None)


REDUCTIONS = {"drop": remove_drop, "renorm": remove_renorm, "kron": remove_kron}


# --------------------------------------------------------------------------
# Spectre
# --------------------------------------------------------------------------
def laplacian_sym(W: np.ndarray) -> np.ndarray:
    """I - D^-1/2 W D^-1/2, symetrise contre la derive numerique."""
    d = W.sum(1)
    inv = np.where(d > EPS, 1.0 / np.sqrt(np.maximum(d, EPS)), 0.0)
    L = -W * inv[:, None] * inv[None, :]
    np.fill_diagonal(L, np.diag(L) + 1.0)
    return 0.5 * (L + L.T)


def spectrum(W: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    vals, vecs = np.linalg.eigh(laplacian_sym(W))
    return np.clip(vals, 0.0, None), vecs


def participation(vecs: np.ndarray, sinks: np.ndarray,
                  band: int = LOW_BAND) -> np.ndarray:
    """part_k = somme des u_k(i)^2 sur S, pour les ``band`` premiers modes."""
    b = min(band, vecs.shape[1])
    return (vecs[sinks, :b] ** 2).sum(0)


def ipr(u: np.ndarray) -> float:
    """Inverse participation ratio. 1/ipr = nombre effectif de noeuds."""
    s2 = (u ** 2).sum()
    return float((u ** 4).sum() / max(s2 ** 2, EPS))


def gap_profile(vals: np.ndarray, band: int = LOW_BAND) -> Dict[str, float]:
    """Ou le spectre place son decoupage naturel, dans la partie basse."""
    b = min(band, len(vals) - 1)
    gaps = np.diff(vals[:b + 1])
    if gaps.size == 0:
        return {"gap_index": float("nan"), "gap_value": float("nan")}
    k = int(gaps.argmax())
    return {"gap_index": float(k + 1), "gap_value": float(gaps[k])}


def spectral_report(W: np.ndarray, sinks: np.ndarray,
                    reduction: str = "kron", band: int = LOW_BAND,
                    roles: Optional[np.ndarray] = None) -> Dict[str, float]:
    """Topologie A contre topologie B sur le meme graphe.

    Rend la fraction d'energie spectrale captee par les puits, les deux
    spectres resumes, et le deplacement du gap. Si ``roles`` est fourni, on
    ajoute la part de u_2 qui vit sur le span insere -- le seul chiffre de
    ce rapport qui utilise les etiquettes.
    """
    vals_a, vecs_a = spectrum(W)
    part = participation(vecs_a, sinks, band)
    deg = W.sum(1)
    vol_share = float(deg[sinks].sum() / max(deg.sum(), EPS))
    inc = float(W[sinks].sum() / max(W.sum(), EPS))

    W_b = REDUCTIONS[reduction](W, sinks)
    vals_b, vecs_b = spectrum(W_b)
    keep = np.setdiff1d(np.arange(W.shape[0]), sinks)

    # Alignement du vecteur de Fiedler : u_2 de A restreint aux noeuds gardes,
    # renormalise, contre u_2 de B. 1 = le retrait n'a rien change a la coupe.
    ua = vecs_a[keep, 1]
    ua = ua / max(np.linalg.norm(ua), EPS)
    align = float(abs(ua @ vecs_b[:, 1]))

    ga, gb = gap_profile(vals_a, band), gap_profile(vals_b, band)
    out = {
        "n_nodes": int(W.shape[0]),
        "n_sinks": int(sinks.size),
        "vol_share_sinks": vol_share,
        "edge_share_sinks": inc,
        "part_u2": float(part[1]) if part.size > 1 else float("nan"),
        "part_low_mean": float(part[1:].mean()) if part.size > 1 else float("nan"),
        "part_max_mode": float(np.argmax(part[1:]) + 1) if part.size > 1 else float("nan"),
        "ipr_u2_A": ipr(vecs_a[:, 1]),
        "ipr_u2_B": ipr(vecs_b[:, 1]),
        "lambda2_A": float(vals_a[1]), "lambda2_B": float(vals_b[1]),
        "eigengap_A": float(vals_a[2] - vals_a[1]),
        "eigengap_B": float(vals_b[2] - vals_b[1]),
        "lambda_max_A": float(vals_a[-1]), "lambda_max_B": float(vals_b[-1]),
        "gap_index_A": ga["gap_index"], "gap_index_B": gb["gap_index"],
        "gap_value_A": ga["gap_value"], "gap_value_B": gb["gap_value"],
        "fiedler_align": align,
        "spec_shift": float(np.abs(np.sort(vals_a)[:band]
                                   - np.sort(vals_b)[:band]).mean()),
    }
    if roles is not None:
        inj = roles == ROLE_INJECTION
        out["part_u2_span_A"] = float((vecs_a[inj, 1] ** 2).sum())
        out["part_u2_span_B"] = float((vecs_b[inj[keep], 1] ** 2).sum())
        out["span_share"] = float(inj.mean())
    return out


# --------------------------------------------------------------------------
# Campagne sur le cache
# --------------------------------------------------------------------------
def _filter_template(W: np.ndarray, roles: np.ndarray):
    keep = roles != ROLE_TEMPLATE
    return W[np.ix_(keep, keep)], roles[keep]


def run(cache_dir: str, layer_stride: int = 4, reduction: str = "kron",
        share: float = 0.02, max_pairs: Optional[int] = None,
        per_head: bool = False, head_stride: int = 1
        ) -> Dict[str, List[Dict]]:
    """Parcourt le cache et rend les rapports, condition par condition.

    ``per_head`` est la topologie qui compte pour le paradoxe G1/G2 : les
    AUROC de 0,83 viennent de graphes par tete, pas du graphe moyenne. Un
    puits peut etre invisible apres moyennage des tetes et dominer deux
    tetes sur quatorze.
    """
    from .attn_cache import iter_cache

    out: Dict[str, List[Dict]] = {"injected": [], "control": []}
    n = 0
    for att_i, att_c, roles_i, roles_c, meta in iter_cache(cache_dir):
        n += 1
        layers = range(0, att_i.shape[0], layer_stride)
        for cond, att, roles in (("injected", att_i, roles_i),
                                 ("control", att_c, roles_c)):
            for l in layers:
                Wh = symmetrize(att[l], per_head=per_head)
                heads = (range(0, Wh.shape[0], head_stride) if per_head
                         else [None])
                for h in heads:
                    W = Wh if h is None else Wh[h]
                    W, r = _filter_template(W, roles)
                    s = sink_indices(W, share=share)
                    if W.shape[0] - s.size < 10:
                        continue
                    row = spectral_report(W, s, reduction=reduction, roles=r)
                    row.update(layer=int(l), pair=int(meta["pair"]),
                               head=-1 if h is None else int(h),
                               position=meta["position"])
                    out[cond].append(row)
        print(f"  {n} paires", flush=True)
        if max_pairs and n >= max_pairs:
            break
    return out


def _agg(rows: List[Dict], key: str) -> float:
    v = np.array([r[key] for r in rows if np.isfinite(r.get(key, np.nan))])
    return float(v.mean()) if v.size else float("nan")


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description="puits d'attention : topologie A vs B")
    p.add_argument("cache_dir")
    p.add_argument("--layer-stride", type=int, default=4)
    p.add_argument("--reduction", default="kron", choices=list(REDUCTIONS))
    p.add_argument("--share", type=float, default=0.02)
    p.add_argument("--max-pairs", type=int, default=None)
    p.add_argument("--per-head", action="store_true",
                   help="un graphe par tete : la topologie du resultat a 0,83")
    p.add_argument("--head-stride", type=int, default=1)
    p.add_argument("--csv", default=None)
    args = p.parse_args(argv)

    res = run(args.cache_dir, args.layer_stride, args.reduction,
              share=args.share, max_pairs=args.max_pairs,
              per_head=args.per_head, head_stride=args.head_stride)
    rows = res["injected"] + res["control"]
    if not rows:
        print("aucune mesure", file=sys.stderr)
        return 1

    print(f"\n{len(rows)} graphes ({len(res['injected'])} injectes), "
          f"retrait = {args.reduction}\n")
    print("Ce que les puits possedent")
    print(f"  noeuds puits                     {_agg(rows,'n_sinks'):.2f} "
          f"sur {_agg(rows,'n_nodes'):.0f}")
    print(f"  part du volume                   {_agg(rows,'vol_share_sinks'):.3f}")
    print(f"  part de la masse d'aretes        {_agg(rows,'edge_share_sinks'):.3f}")
    print(f"  energie de u2 sur les puits      {_agg(rows,'part_u2'):.3f}")
    print(f"  energie moyenne des modes bas    {_agg(rows,'part_low_mean'):.3f}")
    print(f"  mode le plus capte               u{_agg(rows,'part_max_mode'):.1f}")
    print(f"  etalement de u2 (1/IPR)  A       "
          f"{1.0/max(_agg(rows,'ipr_u2_A'),EPS):.1f} noeuds")
    print(f"                           B       "
          f"{1.0/max(_agg(rows,'ipr_u2_B'),EPS):.1f} noeuds")

    print("\nCe que le retrait deplace")
    print(f"{'':34}{'A (G2)':>10}{'B (G1)':>10}{'ecart':>10}")
    for lbl, a, b in (("lambda_2", "lambda2_A", "lambda2_B"),
                      ("ecart lambda_3 - lambda_2", "eigengap_A", "eigengap_B"),
                      ("lambda_max", "lambda_max_A", "lambda_max_B"),
                      ("indice du plus grand ecart", "gap_index_A", "gap_index_B"),
                      ("valeur de cet ecart", "gap_value_A", "gap_value_B")):
        va, vb = _agg(rows, a), _agg(rows, b)
        print(f"  {lbl:<32}{va:>10.4f}{vb:>10.4f}{vb-va:>+10.4f}")
    print(f"  {'alignement de u2 A vs B':<32}{_agg(rows,'fiedler_align'):>10.3f}")
    print(f"  {'decalage du bas du spectre':<32}"
          f"{_agg(rows,'spec_shift'):>10.4f}")

    if "part_u2_span_A" in rows[0]:
        inj, ctl = res["injected"], res["control"]
        print("\nOu u2 se localise (etiquettes utilisees, diagnostic seulement)")
        print(f"  part du span dans le prompt      {_agg(inj,'span_share'):.3f}")
        print(f"  energie de u2 sur le span   A    {_agg(inj,'part_u2_span_A'):.3f}"
              f"   (controle {_agg(ctl,'part_u2_span_A'):.3f})")
        print(f"                              B    {_agg(inj,'part_u2_span_B'):.3f}"
              f"   (controle {_agg(ctl,'part_u2_span_B'):.3f})")

    if args.csv:
        import csv as _csv
        os.makedirs(os.path.dirname(args.csv) or ".", exist_ok=True)
        for cond in ("injected", "control"):
            for r in res[cond]:
                r["condition"] = cond
        with open(args.csv, "w", newline="", encoding="utf-8") as fh:
            w = _csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)
        print(f"\n-> {args.csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
