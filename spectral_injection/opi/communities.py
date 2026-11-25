"""H5 : communautes spectrales et composition regionale.

Le clustering est construit **sans aucune etiquette** : ni labels d'attaque,
ni frontieres de l'instruction injectee. Seules les regions U / E / T, qui
viennent de la construction du prompt, servent ensuite a *decrire* la
composition des communautes obtenues.

Le nombre de communautes n'est pas choisi pour separer benin et attaque. La
regle est non supervisee, fixe et declaree :

    K est balaye sur un ensemble pre-declare, et TOUTES les valeurs sont
    rapportees. Aucune n'est selectionnee.

S'y ajoute une regle intrinseque, l'heuristique de l'ecart spectral : le K
qui maximise ``lambda_{k+1} - lambda_k`` sur la partie basse du spectre.
Elle ne regarde que le spectre, jamais les conditions. Sa valeur est
rapportee comme observation, et sa stabilite entre couches et entre
conditions fait partie des mesures.

Aucune hypothese de purete. Les campagnes precedentes ont montre qu'un span
insere ne forme pas de communaute pure : la bipartition par signe du vecteur
de Fiedler produit deux groupes de taille comparable alors que la region
cible est minoritaire. On mesure donc la **composition** des communautes,
pas leur purete.
"""

from typing import Dict, Optional, Sequence, Tuple

import numpy as np

EPS = 1e-12
#: Valeurs de K pre-declarees. Toutes rapportees, aucune selectionnee.
#:
#: Justification, strictement structurelle et posee avant tout resultat :
#:
#:   K = 2  partition minimale, celle que relaxe le vecteur de Fiedler.
#:          C'est le decoupage le plus grossier qui existe.
#:   K = 3  resolution d'analyse motivee par les trois regions U / E / T.
#:          **Ce n'est pas une hypothese sur le resultat.** On ne presuppose
#:          en aucune facon que le clustering retrouvera ces regions, ni
#:          qu'il doive les retrouver : K = 3 fixe seulement l'echelle a
#:          laquelle on regarde, pour que la composition regionale soit
#:          mesuree a une resolution comparable au nombre de regions. Un
#:          decoupage qui ignore completement les frontieres U / E / T est
#:          un resultat parfaitement admissible a ce K.
#:   K = 5  regime intermediaire, au-dela du nombre de regions.
#:   K = 8  structure fine ; au-dela, les communautes deviennent plus petites
#:          que la region U typique (12 a 46 tokens mesures), et la
#:          composition regionale perd son sens.
#:
#: Aucune de ces valeurs n'a ete choisie en regardant C3 ou C4. Toutes sont
#: rapportees ; aucune n'est designee comme la bonne.
K_VALUES: Tuple[int, ...] = (2, 3, 5, 8)
#: Profondeur du balayage pour l'heuristique de l'ecart spectral.
GAP_SEARCH_MAX = 12


def spectral_embedding(vals: np.ndarray, vecs: np.ndarray, k: int,
                       degrees_: Optional[np.ndarray] = None,
                       kind: str = "shimalik") -> np.ndarray:
    """Plongement spectral a partir d'une decomposition deja calculee.

    ``shimalik`` : ``D^{-1/2} [u_1 .. u_k]``. C'est le plongement associe au
    laplacien symetrique normalise -- celui qui rend le k-means equivalent a
    une relaxation de la coupe normalisee. Le vecteur trivial est conserve :
    sous ``D^{-1/2}`` il devient constant et n'influence donc pas les
    distances relatives entre noeuds, mais le retirer changerait l'echelle
    des autres colonnes.
    """
    U = vecs[:, :k]
    if kind == "shimalik":
        if degrees_ is None:
            raise ValueError("degres requis pour le plongement shimalik")
        inv = np.where(degrees_ > EPS, 1.0 / np.sqrt(np.maximum(degrees_, EPS)), 0.0)
        U = U * inv[:, None]
    elif kind == "njw":
        n = np.linalg.norm(U, axis=1, keepdims=True)
        U = U / np.maximum(n, EPS)
    return np.ascontiguousarray(U)


def kmeanspp(X: np.ndarray, k: int, seed: int = 0, restarts: int = 5,
             iters: int = 100) -> np.ndarray:
    """k-means++ deterministe a graine fixe, clusters vides reinitialises.

    Les clusters vides sont reaffectes au point le plus eloigne de son
    centre : sans ca, un cluster vide fait silencieusement tomber K a K-1 et
    biaise toute comparaison entre valeurs de K.
    """
    n = X.shape[0]
    k = min(k, n)
    rng = np.random.default_rng(seed)
    xn = (X ** 2).sum(1)                      # calcule une fois

    def sqdist(C):
        """||x - c||^2 par produit matriciel : passe par BLAS au lieu
        d'allouer un tableau [n, k, d] a chaque appel."""
        return np.maximum(xn[:, None] - 2.0 * (X @ C.T) + (C ** 2).sum(1), 0.0)

    best, best_cost = None, np.inf
    for _ in range(restarts):
        C = np.empty((k, X.shape[1]))
        C[0] = X[rng.integers(n)]
        for j in range(1, k):
            d2 = sqdist(C[:j]).min(1)
            s = d2.sum()
            p = d2 / s if s > EPS else np.full(n, 1.0 / n)
            C[j] = X[rng.choice(n, p=p)]
        lab = np.full(n, -1, dtype=np.int64)
        for _ in range(iters):
            dist = sqdist(C)
            new = dist.argmin(1)
            if np.array_equal(new, lab):
                break
            lab = new
            empty = []
            for j in range(k):
                m = lab == j
                if m.any():
                    C[j] = X[m].mean(0)
                else:
                    empty.append(j)
            for j in empty:
                far = int(sqdist(C).min(1).argmax())
                C[j] = X[far]
        cost = float(np.take_along_axis(sqdist(C), lab[:, None], 1).sum())
        if cost < best_cost:
            best, best_cost = lab.copy(), cost
    return best


def eigengap_k(vals: np.ndarray, kmax: int = GAP_SEARCH_MAX) -> int:
    """K suggere par l'heuristique de l'ecart spectral. Non supervise.

    **Role dans le protocole.** Cette valeur est une OBSERVATION, pas un
    choix : elle n'est jamais utilisee pour decider sur quel K rapporter les
    resultats. Les K rapportes restent ceux de ``K_VALUES``. Ce que
    ``eigengap_k`` mesure, c'est ce que le spectre lui-meme suggere comme
    nombre de communautes, et sa variation entre couches, entre conditions
    et entre familles d'attaque est une quantite d'interet a part entiere.

    **K = 1 est une sortie legitime** : le plus grand ecart du bas du spectre
    se trouve entre lambda_1 et lambda_2. La valeur est rendue telle quelle,
    sans etre repoussee a 2, parce que la corriger masquerait l'information
    qu'elle porte.

    **Ce que K = 1 ne dit pas.** Ce n'est PAS une preuve qu'il n'y a pas de
    communautes. L'heuristique de l'ecart spectral est une regle de pouce,
    pas un test : elle echoue notamment quand les communautes sont de tailles
    tres inegales, quand elles se recouvrent, ou quand un noeud-moyeu --
    ici le puits d'attention, conserve par decision de protocole -- maintient
    une forte connectivite globale et comprime les premiers ecarts. Sur un
    graphe tenu par un hub, K = 1 est donc le resultat attendu meme si une
    structure communautaire existe en dessous. C'est precisement pourquoi les
    K de ``K_VALUES`` sont tous rapportes independamment de cette valeur.
    """
    m = min(kmax, len(vals) - 1)
    if m < 2:
        return 1
    gaps = np.diff(vals[:m + 1])
    return int(np.argmax(gaps)) + 1


def _entropy(counts: np.ndarray, base: Optional[int] = None) -> float:
    """Entropie de Shannon normalisee par ``log(base)``.

    ``base`` est le nombre de categories **possibles**, pas le nombre de
    categories presentes. La distinction est le defaut qu'il fallait
    corriger : normaliser par le nombre de categories non vides rend les
    valeurs incomparables d'une communaute a l'autre.

    Exemple. Une communaute {U:50, E:50, T:0} et une communaute
    {U:33, E:33, T:33} donnaient toutes deux 1,0 avec l'ancienne
    normalisation -- la premiere divisee par log 2, la seconde par log 3 --
    alors que la seconde est strictement plus melangee. Avec ``base=3``
    elles valent respectivement 0,631 et 1,000.

    Une region absente contribue donc bien 0 a l'entropie et abaisse la
    valeur normalisee, ce qui est le comportement voulu.
    """
    c = np.asarray(counts, dtype=np.float64)
    t = c.sum()
    if t <= 0:
        return float("nan")
    k = int(base) if base is not None else int(c.size)
    if k < 2:
        return 0.0
    p = c[c > 0] / t
    return float(-(p * np.log(p)).sum() / np.log(k))


def composition(labels: np.ndarray, regions: np.ndarray,
                k_requested: Optional[int] = None) -> Dict[str, float]:
    """Composition regionale des communautes. Aucune etiquette d'attaque.

    **Normalisations.** L'entropie regionale est divisee par ``log 3`` -- le
    nombre de regions possibles -- et non par le nombre de regions presentes,
    de sorte que deux communautes soient comparables meme si l'une ne
    contient aucun token technique. La dispersion d'une region est divisee
    par ``log K`` avec ``K`` le nombre de communautes **demandees**, et non
    le nombre de communautes non vides : sans ca, les valeurs ne seraient
    pas comparables entre K, puisque ``log K`` croit avec K.

    ``mix_UE`` est la part des communautes contenant a la fois des tokens de
    U et de E. Aucune hypothese de purete.

    Cas degenere ``K = 1`` : une seule communaute contient tout. L'entropie
    des tailles vaut 0, la dispersion de chaque region vaut 0, et
    ``mix_UE`` vaut 1 puisque l'unique communaute contient bien U et E. Ces
    valeurs sont correctes mais sans contenu informatif ; elles sont rendues
    telles quelles et signalees par ``degenerate_k1``.
    """
    out: Dict[str, float] = {}
    uniq = np.unique(labels)
    regs = ("U", "E", "T")
    n_reg = len(regs)
    K = int(k_requested) if k_requested is not None else int(len(uniq))
    sizes = np.array([(labels == c).sum() for c in uniq], dtype=float)
    out["n_clusters_effective"] = float(len(uniq))
    out["k_requested"] = float(K)
    out["degenerate_k1"] = float(K <= 1 or len(uniq) <= 1)
    out["cluster_size_entropy"] = _entropy(sizes, base=K)
    out["cluster_size_max_share"] = float(sizes.max() / max(sizes.sum(), EPS))

    ent_reg, mixed, dom = [], 0, {r: 0 for r in regs}
    for c in uniq:
        m = labels == c
        cnt = np.array([(regions[m] == r).sum() for r in regs], dtype=float)
        ent_reg.append(_entropy(cnt, base=n_reg))      # toujours log 3
        if cnt[0] > 0 and cnt[1] > 0:
            mixed += 1
        dom[regs[int(cnt.argmax())]] += 1
    out["region_entropy_mean"] = float(np.mean(ent_reg))
    out["region_entropy_max"] = float(np.max(ent_reg))
    out["mix_UE"] = float(mixed / max(len(uniq), 1))
    for r in regs:
        out[f"n_clusters_dominated_by_{r}"] = float(dom[r])

    # dispersion d'une region entre communautes, normalisee par log K
    for r in regs:
        idx = regions == r
        if idx.sum() == 0:
            out[f"spread_{r}"] = float("nan")
            continue
        cnt = np.array([(labels[idx] == c).sum() for c in uniq], dtype=float)
        out[f"spread_{r}"] = _entropy(cnt, base=K)
    return out


def ari(a: np.ndarray, b: np.ndarray) -> float:
    """Indice de Rand ajuste entre deux partitions quelconques.

    Table de contingence par ``bincount`` sur l'indice croise : une seule
    passe au lieu d'une double boucle Python sur les etiquettes. Sur le
    smoke test, cette boucle representait environ un tiers du temps total.
    """
    from math import comb
    ua, ia = np.unique(a, return_inverse=True)
    ub, ib = np.unique(b, return_inverse=True)
    tab = np.bincount(ia * len(ub) + ib,
                      minlength=len(ua) * len(ub)).reshape(len(ua), len(ub))
    n = a.size
    s = sum(comb(int(v), 2) for v in tab.ravel())
    sa = sum(comb(int(v), 2) for v in tab.sum(1))
    sb = sum(comb(int(v), 2) for v in tab.sum(0))
    tot = comb(n, 2)
    if tot == 0:
        return float("nan")
    exp = sa * sb / tot
    mx = 0.5 * (sa + sb)
    return 0.0 if abs(mx - exp) < EPS else float((s - exp) / (mx - exp))


def h5_communities(vals: np.ndarray, vecs: np.ndarray, degrees_: np.ndarray,
                   regions: np.ndarray, k_values: Sequence[int] = K_VALUES,
                   seed: int = 0) -> Tuple[Dict[str, float], Dict[int, np.ndarray]]:
    """Clustering spectral a K declare, et composition regionale.

    Rend (metriques aplaties, partitions par K). Les partitions servent a
    H6 pour mesurer la stabilite entre couches.
    """
    out: Dict[str, float] = {}
    parts: Dict[int, np.ndarray] = {}
    kg = eigengap_k(vals)
    out["k_eigengap"] = float(kg)
    for k in k_values:
        X = spectral_embedding(vals, vecs, k, degrees_)
        lab = kmeanspp(X, k, seed=seed)
        parts[k] = lab
        for key, v in composition(lab, regions, k_requested=k).items():
            out[f"k{k}_{key}"] = v
    return out, parts
