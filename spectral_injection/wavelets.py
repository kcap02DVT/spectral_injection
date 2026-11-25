"""Tache 2 : ondelettes spectrales sur le graphe d'attention (SGWT).

Pourquoi quitter la transformee de Fourier globale. lambda_2 est un scalaire
qui resume tout le graphe ; le span injecte fait 3 a 4 % des tokens. On
demande donc a une statistique globale de refleter une perturbation locale,
et le rapport signal/bruit s'effondre mecaniquement avec T. La transformee
en ondelettes sur graphe (Hammond, Vandergheynst, Gribonval 2011) resout
exactement ce probleme : un atome psi_{s,i} est localise *a la fois* en
frequence (par le noyau g) et dans le graphe (autour du noeud i), et il
donne un score par noeud au lieu d'un scalaire par prompt.

Le noyau. g est passe-bande avec g(0) = 0, donc l'ondelette est aveugle a la
composante constante -- celle qui porte le volume et le puits. C'est la
propriete qui rend ce score structurellement moins sensible aux hubs que
lambda_2, et c'est testable contre la tache 1.

    mexican_hat  g(x) = x exp(1 - x)        passe-bande, g(1) = 1
    heat         g(x) = exp(-x)             passe-bas, pour comparaison
    diff         g(x) = exp(-x) - exp(-2x)  difference de noyaux de chaleur

L'operateur. Psi_s = g(sL), avec L = I - D^-1/2 W D^-1/2 donc spec(L) dans
[0, 2] : lmax = 2 est une borne exacte, pas une estimation, ce qui evite
l'etape de puissance habituelle du SGWT.

Pourquoi Tchebychev. Calculer g(sL) par decomposition propre coute O(T^3) et
annule l'interet de la localisation. L'approximation polynomiale

    g(sL) ~ c_0/2 T_0(Lt) + somme_{k=1..K} c_k T_k(Lt),   Lt = (2/lmax) L - I

se calcule par la recurrence a trois termes, donc en K produits matrice-
vecteur seulement. Et surtout : un polynome de degre K en L est strictement
support dans la boule de rayon K du graphe, ce qui donne a la localisation
spatiale une garantie exacte et non asymptotique.

Le score. Sans signal a filtrer, on filtre les diracs : psi_{s,i} = g(sL)
delta_i, et l'energie ||psi_{s,i}||^2 = [g(sL)^2]_{ii} mesure la structure
locale autour de i a l'echelle s. Aucun apprentissage, aucune etiquette,
aucun signal exterieur -- seulement le graphe.

    compute_local_spectral_energy(L, signal_token, scale)  -> l'energie
    wavelet_node_energy(L, scales)                         -> [S, T] tous noeuds
    window_score(energy, width)                            -> detecteur aveugle

Comparaison avec ``cohesion``. cohesion est une moyenne de bloc a l'interieur
du span, donc elle a besoin du span. window_score prend le maximum sur toutes
les fenetres contigues : elle decouvre la fenetre, et sa position est un
sous-produit. C'est la meme quantite spatiale, sans l'etiquette.

    python -m spectral_injection.wavelets cache/attn/MODELE/SPLIT --selftest
    python -m spectral_injection.wavelets cache/attn/MODELE/SPLIT \\
        --layer-stride 4 --width 12
"""

import argparse
import os
import sys
from typing import Callable, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np

from .prompts import ROLE_INJECTION, ROLE_TEMPLATE

EPS = 1e-12
LMAX_SYM = 2.0         # borne exacte du spectre de L_sym


# --------------------------------------------------------------------------
# Noyaux
# --------------------------------------------------------------------------
def mexican_hat(x: np.ndarray) -> np.ndarray:
    """g(x) = x exp(1 - x) : passe-bande, g(0) = 0, maximum en x = 1."""
    return x * np.exp(1.0 - x)


def heat(x: np.ndarray) -> np.ndarray:
    """g(x) = exp(-x) : passe-bas. Ne s'annule pas en 0, donc sensible au
    mode constant -- present pour mesurer ce que la propriete g(0)=0 apporte."""
    return np.exp(-x)


def diff_heat(x: np.ndarray) -> np.ndarray:
    """Difference de deux noyaux de chaleur : passe-bande, sans parametre."""
    return np.exp(-x) - np.exp(-2.0 * x)


KERNELS: Dict[str, Callable[[np.ndarray], np.ndarray]] = {
    "mexican_hat": mexican_hat, "heat": heat, "diff": diff_heat,
}


def log_scales(n_scales: int = 5, lmax: float = LMAX_SYM,
               lmin_factor: float = 40.0) -> np.ndarray:
    """Echelles log-espacees couvrant la bande utile du spectre.

    Convention de Hammond : s_max place le noyau sur lambda_min = lmax/K et
    s_min sur lmax. Ici lmax = 2 exactement, donc les echelles sont
    comparables d'un prompt a l'autre sans recalibrage -- ce que la
    normalisation sym du laplacien avait justement pour but de garantir.
    """
    lmin = lmax / lmin_factor
    return np.exp(np.linspace(np.log(1.0 / lmax), np.log(1.0 / lmin), n_scales))


# --------------------------------------------------------------------------
# Approximation de Tchebychev
# --------------------------------------------------------------------------
def auto_order(scale: float, lmax: float = LMAX_SYM, base: int = 30,
               cap: int = 2000) -> int:
    """Degre suffisant pour l'echelle demandee.

    A l'echelle s le noyau g(s x) varie sur une largeur ~1/s dans x, donc un
    polynome de degre fixe cesse de le representer des que s lmax depasse
    quelques unites. Le selftest le montrait : a s = 20 et K = 30, l'erreur
    du noyau ``diff`` monte a 4e-2, soit l'ordre de grandeur des effets
    mesures -- inacceptable. Le degre croit donc lineairement en s lmax.

    ``cap`` borne la croissance. Un polynome de degre superieur au diametre
    du graphe n'ajoute aucune localisation -- son support est deja le graphe
    entier -- et sans borne une echelle de 1e5 reclamerait un degre de 6e5 et
    34 Gio de quadrature. Ce n'est pas une approximation cachee : au-dela,
    l'echelle elle-meme est sans objet, puisque g(s lambda) se concentre sous
    le plancher spectral du graphe et l'operateur tend vers zero.
    """
    return int(min(cap, max(base, np.ceil(3.0 * scale * lmax) + 10)))


def chebyshev_coeffs(kernel: Union[str, Callable], scale: float,
                     order: Optional[int] = None, lmax: float = LMAX_SYM,
                     n_quad: Optional[int] = None) -> np.ndarray:
    """Coefficients c_0..c_K de g(s . ) sur [0, lmax].

    Quadrature de Tchebychev-Gauss :

        c_k = (2/N) somme_n g(s x(theta_n)) cos(k theta_n),
        theta_n = pi (n + 1/2) / N,   x(theta) = (lmax/2)(cos theta + 1).

    ``order=None`` laisse ``auto_order`` choisir le degre en fonction de
    l'echelle, ce qui est le defaut sain : un degre fixe est faux aux
    grandes echelles. N est pris a 4(K+1) : assez pour que l'erreur de
    quadrature reste sous l'erreur de troncature, ce que ``selftest``
    verifie au lieu de le supposer.
    """
    g = KERNELS[kernel] if isinstance(kernel, str) else kernel
    order = auto_order(scale, lmax) if order is None else int(order)
    N = n_quad or 4 * (order + 1)
    theta = np.pi * (np.arange(N) + 0.5) / N
    x = 0.5 * lmax * (np.cos(theta) + 1.0)
    gx = np.asarray(g(scale * x), dtype=np.float64)
    k = np.arange(order + 1)
    return (2.0 / N) * (gx[None, :] * np.cos(k[:, None] * theta[None, :])).sum(1)


def chebyshev_apply(L, X: np.ndarray, coeffs: np.ndarray,
                    lmax: float = LMAX_SYM) -> np.ndarray:
    """Applique le polynome de Tchebychev de coefficients ``coeffs`` a X.

    ``L`` peut etre dense ou ``scipy.sparse`` : seuls des produits L @ X sont
    utilises, ce qui est tout l'interet de la methode. X est [T] ou [T, m].
    Le cout est K produits, jamais une decomposition propre.
    """
    a = 2.0 / lmax

    def shifted(V):                       # (2/lmax) L V - V
        return a * (L @ V) - V

    T0 = np.asarray(X, dtype=np.float64)
    out = 0.5 * coeffs[0] * T0
    if coeffs.size == 1:
        return out
    T1 = shifted(T0)
    out = out + coeffs[1] * T1
    for c in coeffs[2:]:
        T2 = 2.0 * shifted(T1) - T0
        out = out + c * T2
        T0, T1 = T1, T2
    return out


# --------------------------------------------------------------------------
# Le livrable
# --------------------------------------------------------------------------
def compute_local_spectral_energy(
    L, signal_token: Union[int, np.ndarray, None], scale: float,
    kernel: Union[str, Callable] = "mexican_hat", order: Optional[int] = None,
    lmax: float = LMAX_SYM, window: Optional[Tuple[int, int]] = None,
) -> Dict[str, Union[float, np.ndarray]]:
    """Energie spectrale locale a l'echelle ``scale``.

    ``signal_token``
        ``int``    -> dirac sur ce token : rend l'ondelette psi_{s,i} et son
                      energie, qui dit comment la structure du graphe se
                      comporte autour de ce token.
        ``array``  -> signal de graphe quelconque (une valeur par token) :
                      rend ses coefficients d'ondelette.
        ``None``   -> tous les diracs a la fois : rend l'energie par noeud
                      ``[g(sL)^2]_ii``, soit un score par token sans aucun
                      signal exterieur ni etiquette.

    ``window`` (lo, hi) inclusif : ajoute ``energy_window``, l'energie
    moyenne sur cette fenetre rapportee a la mediane des noeuds. C'est le
    pendant direct de ``cohesion``, calcule sur le graphe au lieu des poids
    bruts.
    """
    c = chebyshev_coeffs(kernel, scale, order, lmax)
    n = L.shape[0]

    if signal_token is None:
        psi = chebyshev_apply(L, np.eye(n), c, lmax)         # [T, T]
        energy = (psi ** 2).sum(0)                           # ||psi_i||^2
        coeff = None
    elif np.isscalar(signal_token):
        delta = np.zeros(n)
        delta[int(signal_token)] = 1.0
        coeff = chebyshev_apply(L, delta, c, lmax)
        energy = coeff ** 2
    else:
        x = np.asarray(signal_token, dtype=np.float64).reshape(n)
        coeff = chebyshev_apply(L, x, c, lmax)
        energy = coeff ** 2

    med = float(np.median(energy))
    out: Dict[str, Union[float, np.ndarray]] = {
        "energy": energy,
        "coeff": coeff,
        "total": float(energy.sum()),
        "median": med,
        "ipr": float((energy ** 2).sum() / max(energy.sum() ** 2, EPS)),
    }
    if window is not None:
        lo, hi = window
        out["energy_window"] = float(energy[lo:hi + 1].mean() / max(med, EPS))
    return out


def wavelet_node_energy(L, scales: Sequence[float],
                        kernel: Union[str, Callable] = "mexican_hat",
                        order: Optional[int] = None, lmax: float = LMAX_SYM,
                        probes: Optional[int] = None,
                        seed: int = 0) -> np.ndarray:
    """Energie par noeud a chaque echelle : [n_scales, T].

    Exact par defaut (le polynome est applique a l'identite, donc K produits
    de matrices). ``probes=m`` passe a l'estimateur de Hutchinson avec des
    sondes de Rademacher, O(m K T^2) au lieu de O(K T^3) : utile au-dela de
    T ~ 2000, inutile en dessous. L'estimateur est non biaise ; sa variance
    decroit en 1/m.
    """
    n = L.shape[0]
    if probes is None:
        X = np.eye(n)
    else:
        rng = np.random.default_rng(seed)
        X = rng.choice([-1.0, 1.0], size=(n, probes)) / np.sqrt(probes)
    out = np.empty((len(scales), n))
    for i, s in enumerate(scales):
        c = chebyshev_coeffs(kernel, float(s), order, lmax)
        P = chebyshev_apply(L, X, c, lmax)
        out[i] = (P ** 2).sum(1) if probes is not None else (P ** 2).sum(0)
    return out


def robust_z(energy: np.ndarray) -> np.ndarray:
    """(e - mediane) / MAD. La MAD plutot que l'ecart-type parce qu'une
    anomalie de 4 % des noeuds gonfle l'ecart-type qu'elle est censee
    depasser."""
    med = np.median(energy)
    mad = np.median(np.abs(energy - med))
    return (energy - med) / max(1.4826 * mad, EPS)


def window_score(energy: np.ndarray, width: int = 12,
                 positions: Optional[np.ndarray] = None
                 ) -> Dict[str, Union[float, np.ndarray]]:
    """Detecteur aveugle : meilleure fenetre contigue de ``width`` tokens.

    Aucune etiquette. La moyenne glissante est calculee par somme cumulee,
    donc O(T). On rend le score, la position de la fenetre, et un z par
    noeud -- ce dernier est ce qui permet d'exciser le span au lieu de
    seulement lever un drapeau.
    """
    z = robust_z(energy)
    w = min(width, z.size)
    c = np.concatenate([[0.0], np.cumsum(z)])
    means = (c[w:] - c[:-w]) / w
    k = int(means.argmax())
    pos = positions if positions is not None else np.arange(z.size)
    return {"score": float(means[k]), "start": int(pos[k]),
            "end": int(pos[k + w - 1]), "z": z,
            "peak": float(z.max())}


def span_overlap(start: int, end: int, roles: np.ndarray) -> float:
    """Jaccard entre la fenetre decouverte et le span insere. Diagnostic
    seulement : c'est la seule fonction du module qui lit les etiquettes."""
    inj = np.flatnonzero(roles == ROLE_INJECTION)
    if inj.size == 0:
        return float("nan")
    win = np.arange(start, end + 1)
    inter = np.intersect1d(win, inj).size
    union = np.union1d(win, inj).size
    return float(inter / max(union, 1))


# --------------------------------------------------------------------------
# Graphe
# --------------------------------------------------------------------------
def sparsify(W: np.ndarray, mode: str) -> np.ndarray:
    """Sparsification, identique a celle de ``h10_vec.threshold``.

    Indispensable, et pas un reglage : l'attention dense est presque un
    graphe complet, ou tout noeud ressemble a tout noeud. Mesure sur le
    cache, l'energie d'ondelette y est uniforme a 1/T pres (IPR = 0,0041 =
    1/244), donc il n'y a rien a localiser. L'ondelette a besoin d'un
    voisinage, et un voisinage demande des aretes absentes.
    """
    if mode == "dense":
        return W
    A = np.array(W, dtype=np.float64, copy=True)
    np.fill_diagonal(A, 0.0)
    if mode.startswith("q"):
        off = A[A > 0]
        if off.size == 0:
            return A
        return np.where(A >= np.quantile(off, int(mode[1:]) / 100.0), A, 0.0)
    if mode.startswith("topk"):
        k = int(mode[4:])
        if A.shape[0] <= k:
            return A
        thr = np.partition(A, -k, axis=-1)[:, -k][:, None]
        M = A >= thr
        return np.where(M | M.T, A, 0.0)
    raise ValueError(f"sparsification inconnue : {mode}")


def graph_from_attention(attn_layer: np.ndarray, roles: np.ndarray,
                         reduction: str = "none", share: float = 0.02,
                         head: Optional[int] = None,
                         sparse: str = "dense"):
    """[H,T,T] -> (L_sym, roles gardes, positions d'origine).

    ``head=None`` moyenne les tetes ; un entier prend cette tete seule. La
    distinction n'est pas cosmetique : sur le cache, u_2 s'etale sur 80
    noeuds apres moyennage et sur 10 par tete. Le graphe moyenne n'a pas de
    structure locale a offrir a une ondelette, le graphe par tete en a.

    ``reduction`` reprend les trois retraits de la tache 1, ``sparse`` les
    seuillages de la tache 3 : le score d'ondelette est donc mesurable sur
    n'importe laquelle des topologies du depot sans rien recoder.
    """
    from .sinks import (REDUCTIONS, laplacian_sym, sink_indices, symmetrize)

    W = symmetrize(attn_layer, per_head=head is not None)
    if head is not None:
        W = W[head]
    keep = roles != ROLE_TEMPLATE
    W = W[np.ix_(keep, keep)]
    r = roles[keep]
    pos = np.flatnonzero(keep)
    if reduction != "none":
        s = sink_indices(W, share=share)
        kept = np.setdiff1d(np.arange(W.shape[0]), s)
        W = REDUCTIONS[reduction](W, s)
        r, pos = r[kept], pos[kept]
    return laplacian_sym(sparsify(W, sparse)), r, pos


# --------------------------------------------------------------------------
# Verification de l'approximation
# --------------------------------------------------------------------------
def selftest(n: int = 120, order: Optional[int] = None, seed: int = 0) -> int:
    """Tchebychev contre decomposition propre exacte, sur un graphe planté.

    Si ce test echoue, tout le reste du module est faux : on compare
    g(sL) x calcule par le polynome et par la base propre, sur un graphe
    dont on connait la structure.
    """
    rng = np.random.default_rng(seed)
    W = rng.random((n, n)) * 0.05
    W[:30, :30] += 0.6                       # un bloc dense, l'analogue du span
    W = 0.5 * (W + W.T)
    np.fill_diagonal(W, 0.0)
    from .sinks import laplacian_sym
    L = laplacian_sym(W)
    vals, vecs = np.linalg.eigh(L)
    x = rng.standard_normal(n)

    print(f"graphe test : {n} noeuds, lambda_max = {vals[-1]:.4f} "
          f"(borne theorique 2)")
    worst = 0.0
    for name in KERNELS:
        for s in log_scales(4):
            c = chebyshev_coeffs(name, s, order)
            approx = chebyshev_apply(L, x, c)
            exact = vecs @ (KERNELS[name](s * vals) * (vecs.T @ x))
            err = np.abs(approx - exact).max() / max(np.abs(exact).max(), EPS)
            worst = max(worst, err)
            print(f"  {name:<12} s={s:6.3f}  erreur relative max {err:.2e}")

    # L'energie par noeud, exacte contre polynomiale.
    e_poly = wavelet_node_energy(L, [0.5], order=order)[0]
    G = vecs @ np.diag(mexican_hat(0.5 * vals)) @ vecs.T
    e_exact = (G ** 2).sum(0)
    err_e = np.abs(e_poly - e_exact).max() / max(e_exact.max(), EPS)
    print(f"  energie par noeud : erreur relative max {err_e:.2e}")

    # Le support d'un polynome de degre K est exactement la boule de rayon K.
    # Il faut un graphe de diametre superieur a K pour que l'enonce ait un
    # contenu : sur le graphe aleatoire ci-dessus, la boule de rayon 4 est
    # deja tout le graphe et le test serait vide. D'ou une chaine.
    K = 4
    P = np.zeros((40, 40))
    i = np.arange(39)
    P[i, i + 1] = P[i + 1, i] = 1.0
    Lp = laplacian_sym(P)
    c = chebyshev_coeffs("mexican_hat", 1.0, order=K)
    psi = chebyshev_apply(Lp, np.eye(40)[:, 0], c)
    inside = np.abs(psi[:K + 1]).max()
    leak = np.abs(psi[K + 1:]).max()
    print(f"  chaine de 40 noeuds, polynome de degre {K} :")
    print(f"    amplitude dans la boule de rayon {K} : {inside:.3e}")
    print(f"    fuite au-dela                        : {leak:.3e}")

    ok = worst < 1e-6 and err_e < 1e-6 and leak < 1e-12
    print("\n" + ("OK : l'approximation est exacte au niveau machine."
                  if ok else "ECHEC : augmenter --order."))
    return 0 if ok else 1


# --------------------------------------------------------------------------
# Campagne sur le cache
# --------------------------------------------------------------------------
def run(cache_dir: str, layer_stride: int = 4, width: int = 12,
        kernel: str = "mexican_hat", order: Optional[int] = None,
        reduction: str = "none", n_scales: int = 5,
        max_pairs: Optional[int] = None, per_head: bool = False,
        head_stride: int = 1, sparse: str = "dense") -> List[Dict]:
    """Un enregistrement par (paire, couche, tete, echelle, condition)."""
    from .attn_cache import iter_cache

    scales = log_scales(n_scales)
    rows: List[Dict] = []
    n = 0
    for att_i, att_c, roles_i, roles_c, meta in iter_cache(cache_dir):
        n += 1
        n_heads = att_i.shape[1]
        heads = list(range(0, n_heads, head_stride)) if per_head else [None]
        for l in range(0, att_i.shape[0], layer_stride):
            for h in heads:
                for cond, att, roles in (("injected", att_i, roles_i),
                                         ("control", att_c, roles_c)):
                    L, r, pos = graph_from_attention(att[l], roles, reduction,
                                                     head=h, sparse=sparse)
                    inj = np.flatnonzero(r == ROLE_INJECTION)
                    if inj.size < 2:
                        continue
                    span = (int(inj[0]), int(inj[-1]))
                    E = wavelet_node_energy(L, scales, kernel, order)
                    for si, s in enumerate(scales):
                        w = window_score(E[si], width, positions=pos)
                        med = max(float(np.median(E[si])), EPS)
                        rows.append({
                            "pair": int(meta["pair"]), "layer": int(l),
                            "head": -1 if h is None else int(h),
                            "scale": float(s), "condition": cond,
                            "position": meta["position"],
                            "window_score": w["score"], "peak_z": w["peak"],
                            "overlap": span_overlap(w["start"], w["end"], r)
                            if cond == "injected" else float("nan"),
                            # pendant direct de cohesion : energie sur le span
                            "span_energy": float(
                                E[si][span[0]:span[1] + 1].mean() / med),
                            "ipr": float((E[si] ** 2).sum()
                                         / max(E[si].sum() ** 2, EPS)),
                        })
        print(f"  {n} paires", flush=True)
        if max_pairs and n >= max_pairs:
            break
    return rows


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description="ondelettes spectrales sur le graphe")
    p.add_argument("cache_dir", nargs="?")
    p.add_argument("--selftest", action="store_true")
    p.add_argument("--layer-stride", type=int, default=4)
    p.add_argument("--width", type=int, default=12)
    p.add_argument("--kernel", default="mexican_hat", choices=list(KERNELS))
    p.add_argument("--order", type=int, default=None,
                   help="degre du polynome (defaut : adapte a l'echelle)")
    p.add_argument("--n-scales", type=int, default=5)
    p.add_argument("--reduction", default="none",
                   choices=["none", "drop", "renorm", "kron"])
    p.add_argument("--per-head", action="store_true",
                   help="un graphe par tete : la seule topologie ou u2 est "
                        "localise (10 noeuds contre 80 apres moyennage)")
    p.add_argument("--head-stride", type=int, default=1)
    p.add_argument("--sparse", default="dense",
                   help="dense, q80/q90/q95, topk16 : l'attention dense est "
                        "quasi complete, l'ondelette n'y voit aucun voisinage")
    p.add_argument("--max-pairs", type=int, default=None)
    p.add_argument("--csv", default=None)
    args = p.parse_args(argv)

    if args.selftest:
        return selftest(order=args.order)
    if not args.cache_dir:
        print("il faut un cache, ou --selftest", file=sys.stderr)
        return 2

    rows = run(args.cache_dir, args.layer_stride, args.width, args.kernel,
               args.order, args.reduction, args.n_scales, args.max_pairs,
               args.per_head, args.head_stride, args.sparse)
    if not rows:
        print("aucune mesure", file=sys.stderr)
        return 1

    from .hypotheses_report import auroc

    print(f"\n{len(rows)} mesures · noyau {args.kernel} · fenetre {args.width} "
          f"tokens · retrait {args.reduction}\n")
    print(f"{'echelle':>9}{'couche':>8}{'tete':>6}{'n':>5}"
          f"{'AUROC fenetre':>15}{'AUROC span':>12}"
          f"{'recouvr.':>10}{'IPR inj':>9}")
    print("-" * 74)
    scales = sorted({r["scale"] for r in rows})
    cells = sorted({(r["layer"], r["head"]) for r in rows})
    best = (0.0, None)
    for s in scales:
        for l, h in cells:
            sub = [r for r in rows if r["scale"] == s and r["layer"] == l
                   and r["head"] == h]
            inj = {r["pair"]: r for r in sub if r["condition"] == "injected"}
            ctl = {r["pair"]: r for r in sub if r["condition"] == "control"}
            keys = sorted(set(inj) & set(ctl))
            if len(keys) < 5:
                continue
            lab = np.r_[np.ones(len(keys)), np.zeros(len(keys))]
            a_win = auroc(lab, np.r_[[inj[k]["window_score"] for k in keys],
                                     [ctl[k]["window_score"] for k in keys]])
            a_span = auroc(lab, np.r_[[inj[k]["span_energy"] for k in keys],
                                      [ctl[k]["span_energy"] for k in keys]])
            ov = float(np.nanmean([inj[k]["overlap"] for k in keys]))
            ip = float(np.mean([inj[k]["ipr"] for k in keys]))
            print(f"{s:>9.3f}{l:>8}{h:>6}{len(keys):>5}{a_win:>15.3f}"
                  f"{a_span:>12.3f}{ov:>10.3f}{ip:>9.4f}")
            if max(a_win, 1 - a_win) > best[0]:
                best = (max(a_win, 1 - a_win), (s, l, h, a_win))
    print()
    if best[1]:
        s, l, h, a = best[1]
        print(f"Meilleure cellule aveugle : echelle {s:.3f}, couche {l}, "
              f"tete {h}, AUROC {a:.3f}")
    print("AUROC fenetre = detecteur aveugle (la fenetre est decouverte).")
    print("AUROC span    = meme energie, lue sur le span connu : c'est le")
    print("                pendant de cohesion, et la borne haute de l'autre.")
    print("Ces maxima sont pris sur echelles x couches : a confirmer sur un")
    print("split a cellule fixee avant d'etre compares a 0,86.")

    if args.csv:
        import csv as _csv
        os.makedirs(os.path.dirname(args.csv) or ".", exist_ok=True)
        with open(args.csv, "w", newline="", encoding="utf-8") as fh:
            w = _csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)
        print(f"\n-> {args.csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
