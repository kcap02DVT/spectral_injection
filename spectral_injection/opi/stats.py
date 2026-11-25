"""Contrastes apparies entre conditions, sans aucun apprentissage.

L'unite statistique est **l'exemple** -- une ligne du dataset. Les couches et
les tetes mesurees sur un meme exemple sont des observations dependantes :
elles ne comptent jamais comme des echantillons separes. Elles sont donc
agregees en un scalaire par exemple AVANT toute inference, selon une regle
preespecifiee.

Contraste principal :

    Delta M_i = M_i(C4) - M_i(C3)

**Portee de ce contraste.** Il mesure l'effet du remplacement d'un bloc de
meme longueur, pas l'effet causal de la malveillance. C3 et C4 different
simultanement par la forme syntaxique (declarative contre imperative), le
destinataire (le lecteur contre le modele), la presence d'une tache
concurrente, le contenu lexical et la coherence semantique avec l'hote.
Une difference significative etablit qu'il se passe quelque chose, pas
lequel de ces facteurs le produit. Voir ``confounders.py`` pour le detail.

Hypothese testee et ce qu'elle suppose. Le test par inversion de signe
repose sur l'**echangeabilite sous H0** : si C3 et C4 ne different pas, la
difference appariee ``Delta M_i`` a une distribution symetrique autour de
zero, et changer son signe laisse cette distribution inchangee. C'est plus
faible que la normalite, mais plus fort que << mediane nulle >> : une
asymetrie reelle sous H0 invaliderait le test. L'appariement intra-exemple
rend l'hypothese plausible ici, puisque C3 et C4 partagent le meme document
hote, la meme instruction, le meme prefixe d'attaque et la meme longueur en
tokens -- seul le bloc remplace differe.

Les permutations se font **au niveau de l'exemple** : inverser un exemple
inverse ensemble toutes ses couches et toutes ses tetes. C'est ce qui
preserve la dependance interne et rend le maximum sur la famille un seuil
familial valide.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

EPS = 1e-12

#: Agregation preespecifiee des couches et des tetes vers un scalaire par
#: exemple. La mediane est retenue pour l'inference principale : elle ne
#: privilegie aucune couche ni aucune tete, et resiste a une poignee de
#: cellules aberrantes. La moyenne est calculee aussi, comme analyse
#: secondaire -- jamais choisie apres coup en fonction du resultat.
AGGREGATORS = {
    "median": lambda x, ax: np.nanmedian(x, axis=ax),
    "mean": lambda x, ax: np.nanmean(x, axis=ax),
}
PRIMARY_AGGREGATOR = "median"


# ==========================================================================
# Famille confirmatoire, PRE-DECLAREE
# ==========================================================================
#: Deux metriques par hypothese, choisies sur des bases structurelles avant
#: tout calcul de contraste. Aucune n'a ete retenue pour son pouvoir de
#: separation entre C3 et C4.
#:
#: H1  lambda_2        la connectivite algebrique, quantite centrale de H1
#:     eigengap_23     l'ecart spectral, qui dit si un decoupage en deux est
#:                     structurellement marque
#: H2  conductance     la frontiere de E rapportee au plus petit volume
#:     cut_over_vol_E  la meme coupe rapportee au seul volume de E, qui ne
#:                     bascule pas de denominateur quand E devient majoritaire
#: H3  Fbar_E_to_U     le flux normalise de E vers l'instruction legitime,
#:                     la question explicite de H3
#:     Fbar_E_to_E     le flux interne a E, son terme de comparaison direct
#: H4  h4B_hfer        la part de haute frequence du signal indicateur,
#:                     variante B (projetee hors du noyau spectral)
#:     h4B_centroid    le centroide spectral de ce meme signal
#: H5  k3_mix_UE       le melange U/E a la resolution des trois regions
#:     k3_region_entropy_mean  l'entropie de composition regionale a ce K
#: H6  traj lambda_2 total_variation   l'amplitude du parcours en profondeur
#:     traj h4B_hfer total_variation   idem pour le signal de region
CONFIRMATORY: Tuple[str, ...] = (
    "h1_lambda_2", "h1_eigengap_23",
    "h2_conductance", "h2_cut_over_vol_E",
    "h3_Fbar_E_to_U", "h3_Fbar_E_to_E",
    "h4B_hfer", "h4B_centroid",
    "h5_k3_mix_UE", "h5_k3_region_entropy_mean",
)
CONFIRMATORY_TRAJ: Tuple[str, ...] = (
    "traj_h1_lambda_2__total_variation",
    "traj_h4B_hfer__total_variation",
)

#: Comparaisons secondaires. Elles ne changent pas le statut principal de
#: C4 - C3 et ne sont pas corrigees dans la meme famille.
SECONDARY_CONTRASTS = (("C4", "C2"), ("C4", "C1"), ("C2", "C1"), ("C3", "C1"))


# ==========================================================================
# Agregation exemple par exemple
# ==========================================================================
def aggregate_examples(V: np.ndarray, how: str = PRIMARY_AGGREGATOR
                       ) -> np.ndarray:
    """``[n, cond, couches, tetes, metriques]`` -> ``[n, cond, metriques]``.

    Les axes des couches et des tetes sont reduits ensemble : ce sont des
    observations dependantes d'un meme exemple, pas des echantillons.
    """
    V = np.asarray(V, dtype=np.float64)
    if V.ndim != 5:
        raise ValueError(f"attendu [n, cond, L, H, M], recu {V.shape}")
    with np.errstate(invalid="ignore"):
        return AGGREGATORS[how](V, (2, 3))


def aggregate_trajectories(T: np.ndarray, how: str = PRIMARY_AGGREGATOR
                           ) -> np.ndarray:
    """``[n, cond, tetes, stats]`` -> ``[n, cond, stats]``.

    Les statistiques de trajectoire sont deja calculees par tete sur toute
    la profondeur ; il ne reste qu'a reduire l'axe des tetes.
    """
    T = np.asarray(T, dtype=np.float64)
    if T.ndim != 4:
        raise ValueError(f"attendu [n, cond, H, S], recu {T.shape}")
    with np.errstate(invalid="ignore"):
        return AGGREGATORS[how](T, 2)


# ==========================================================================
# Differences appariees
# ==========================================================================
@dataclass
class PairedData:
    """Differences appariees pretes pour l'inference."""
    diff: np.ndarray                 # [n_valid, n_metrics]
    names: List[str]
    kept: np.ndarray                 # indices des exemples retenus
    dropped: np.ndarray              # indices ecartes, appariement incomplet
    strata: Optional[np.ndarray] = None
    contrast: Tuple[str, str] = ("C4", "C3")

    @property
    def n(self) -> int:
        return int(self.diff.shape[0])


def paired_differences(A: np.ndarray, names: Sequence[str],
                       conditions: Sequence[str], a: str = "C4",
                       b: str = "C3", strata: Optional[np.ndarray] = None
                       ) -> PairedData:
    """``A`` ``[n, cond, metriques]`` -> differences ``M(a) - M(b)``.

    **Appariements incomplets.** Un exemple dont l'une des deux conditions
    est manquante -- typiquement C3 lorsque le remplissage benin n'a pas pu
    etre apparie au token -- est ecarte de l'inference et son indice est
    conserve dans ``dropped``. Il n'est jamais impute ni silencieusement
    ignore : le compte des exemples ecartes fait partie du rapport.
    """
    A = np.asarray(A, dtype=np.float64)
    conds = list(conditions)
    ia, ib = conds.index(a), conds.index(b)
    d = A[:, ia, :] - A[:, ib, :]
    ok = np.isfinite(d).all(axis=1)
    kept = np.flatnonzero(ok)
    dropped = np.flatnonzero(~ok)
    return PairedData(diff=d[ok], names=list(names), kept=kept,
                      dropped=dropped,
                      strata=None if strata is None else np.asarray(strata)[ok],
                      contrast=(a, b))


# ==========================================================================
# Tailles d'effet et descriptifs
# ==========================================================================
def describe(pd_: PairedData) -> List[Dict]:
    """Statistiques descriptives des differences, par metrique."""
    D = pd_.diff
    n = D.shape[0]
    out = []
    for j, name in enumerate(pd_.names):
        d = D[:, j]
        sd = d.std(ddof=1) if n > 1 else np.nan
        out.append({
            "metric": name, "n": n,
            "mean": float(d.mean()), "median": float(np.median(d)),
            "sd": float(sd),
            "iqr": float(np.quantile(d, 0.75) - np.quantile(d, 0.25)),
            # taille d'effet appariee de Cohen : moyenne / ecart-type des
            # differences. Sans unite, comparable entre metriques.
            "dz": float(d.mean() / sd) if sd and sd > EPS else np.nan,
            "win_rate": float((d > 0).mean()),
            "n_positive": int((d > 0).sum()),
            "n_zero": int((d == 0).sum()),
        })
    return out


# ==========================================================================
# Permutation par inversion de signe, au niveau de l'exemple
# ==========================================================================
#: Au-dela de ce nombre d'exemples on bascule en Monte-Carlo : 2^n
#: configurations deviennent impraticables.
EXACT_MAX_N = 16


def _standardise(D: np.ndarray) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """Differences ramenees a une echelle commune, sans degenerescence.

    La statistique retenue est ``|moyenne(d / s)|``, ou ``s`` est une echelle
    **fixee sur l'observation** et non recalculee a chaque permutation.

    Pourquoi pas le t de Student recalcule. Avec des differences toutes
    identiques -- disons toutes egales a +5 -- la variance de la
    configuration observee est nulle, donc son t vaut 0 par convention,
    tandis que les configurations permutees ont un t fini. Un effet parfait
    ressortait alors avec p = 1. Le defaut etait present dans la premiere
    version du module.

    L'echelle est choisie ainsi :

        sd > 0            s = sd           cas ordinaire
        sd = 0, moy != 0  s = |moyenne|    differences constantes non nulles :
                          la statistique observee vaut alors exactement 1 et
                          seules les deux configurations extremes l'egalent,
                          ce qui donne p = 2 / 2^n, le plancher exact
        sd = 0, moy = 0   s = 1            metrique entierement constante :
                          statistique nulle partout, p = 1

    Fixer l'echelle ne compromet pas l'exactitude : sous l'hypothese de
    symetrie, toute statistique fonction des donnees et des signes donne un
    test exact, puisque toutes les configurations sont equiprobables.
    """
    mean = D.mean(0)
    sd = D.std(0, ddof=1) if D.shape[0] > 1 else np.zeros(D.shape[1])
    kind = []
    scale = np.empty_like(mean)
    for j in range(D.shape[1]):
        if sd[j] > EPS:
            scale[j], k = sd[j], "ok"
        elif abs(mean[j]) > EPS:
            scale[j], k = abs(mean[j]), "constant_nonzero"
        else:
            scale[j], k = 1.0, "constant_zero"
        kind.append(k)
    return D / scale, scale, kind


def _all_sign_patterns(n: int, chunk: int = 4096):
    """Enumere les 2^n configurations de signes, par blocs."""
    total = 1 << n
    bits = np.arange(n)
    for start in range(0, total, chunk):
        stop = min(start + chunk, total)
        idx = np.arange(start, stop, dtype=np.int64)[:, None]
        yield 1.0 - 2.0 * ((idx >> bits) & 1).astype(np.float64)


def sign_flip_test(pd_: PairedData, n_perm: int = 10000, seed: int = 0,
                   exact_max_n: int = EXACT_MAX_N) -> List[Dict]:
    """Test par inversion de signe, correction max-statistic sur la famille.

    Un tirage inverse le signe d'exemples **entiers** : toutes les couches,
    toutes les tetes et **toutes les metriques de la famille** basculent
    ensemble. La dependance entre metriques est donc integralement
    preservee, ce qui rend le maximum sur la famille un seuil familial
    valide sans hypothese sur la structure de correlation.

    **Enumeration exacte** quand ``n <= exact_max_n`` : les 2^n
    configurations sont parcourues, la configuration observee comprise, et
    la p-valeur est le quotient exact. Aucune correction ``+1`` n'est alors
    appliquee -- elle serait injustifiee puisqu'il n'y a pas
    d'echantillonnage. Au-dela, tirage de Monte-Carlo avec la correction
    ``(count + 1) / (n_perm + 1)``, qui garantit la validite conservative.

    **Planchers de p-valeur**, distingues et rapportes separement :

        p_floor_exact  2 / 2^n = 2^(1-n)
                       Bilateral : la statistique |.| est invariante par
                       inversion globale, donc la valeur extreme est
                       atteinte par au moins DEUX configurations -- toutes
                       les differences gardees, et toutes inversees. Le
                       plancher n'est donc pas 2^-n.
        p_floor_mc     1 / (n_perm + 1), resolution de l'echantillonnage.
        p_floor        le plus grand des deux effectivement applicable.
    """
    D = pd_.diff
    n, m = D.shape
    floor_exact = 2.0 ** (1 - n) if n >= 1 else np.nan
    if n < 2:
        return [{"metric": nm, "stat": np.nan, "p_raw": np.nan,
                 "p_fwer": np.nan, "method": "insufficient_n",
                 "n_perm_effective": 0, "p_floor_exact": floor_exact,
                 "p_floor_mc": np.nan, "p_floor": np.nan,
                 "degenerate": "n<2"} for nm in pd_.names]

    Z, scale, kind = _standardise(D)
    obs = np.abs(Z.mean(0))

    exact = n <= exact_max_n
    ge_raw = np.zeros(m)
    ge_max = np.zeros(m)
    total = 0
    if exact:
        for flips in _all_sign_patterns(n):
            s = np.abs(flips @ Z) / n
            ge_raw += (s >= obs[None, :] - 1e-12).sum(0)
            ge_max += (s.max(1)[:, None] >= obs[None, :] - 1e-12).sum(0)
            total += flips.shape[0]
        p_raw = ge_raw / total
        p_fwer = ge_max / total
        floor_mc = np.nan
        floor = floor_exact
    else:
        rng = np.random.default_rng(seed)
        done = 0
        while done < n_perm:
            b = min(2000, n_perm - done)
            flips = rng.choice([-1.0, 1.0], size=(b, n))
            s = np.abs(flips @ Z) / n
            ge_raw += (s >= obs[None, :] - 1e-12).sum(0)
            ge_max += (s.max(1)[:, None] >= obs[None, :] - 1e-12).sum(0)
            done += b
        total = n_perm
        p_raw = (ge_raw + 1) / (n_perm + 1)
        p_fwer = (ge_max + 1) / (n_perm + 1)
        floor_mc = 1.0 / (n_perm + 1)
        floor = max(floor_exact, floor_mc)

    return [{"metric": pd_.names[j], "stat": float(obs[j]),
             "scale": float(scale[j]), "degenerate": kind[j],
             "p_raw": float(p_raw[j]), "p_fwer": float(p_fwer[j]),
             "method": "exact" if exact else "monte_carlo",
             "n_perm_effective": int(total),
             "p_floor_exact": float(floor_exact),
             "p_floor_mc": float(floor_mc) if floor_mc == floor_mc else np.nan,
             "p_floor": float(floor)}
            for j in range(m)]


# ==========================================================================
# Bootstrap apparie, stratifie
# ==========================================================================
def bootstrap_ci(pd_: PairedData, n_boot: int = 5000, alpha: float = 0.05,
                 seed: int = 0, min_per_stratum: int = 2
                 ) -> List[Dict]:
    """Intervalles percentile sur la moyenne et la mediane des differences.

    Le reechantillonnage porte sur les **exemples**, jamais sur les couches
    ni les tetes. Si ``strata`` est fourni et que chaque strate compte au
    moins ``min_per_stratum`` exemples, le tirage est stratifie : on
    reechantillonne a l'interieur de chaque strate en conservant son
    effectif. Sinon on retombe sur un bootstrap simple, et le rapport le
    signale au lieu de le masquer.
    """
    D = pd_.diff
    n, m = D.shape
    rng = np.random.default_rng(seed)
    strat = pd_.strata
    use_strata = False
    groups: List[np.ndarray] = []
    if strat is not None:
        groups = [np.flatnonzero(strat == s) for s in np.unique(strat)]
        use_strata = all(len(g) >= min_per_stratum for g in groups) and len(groups) > 1

    boot_mean = np.empty((n_boot, m))
    boot_med = np.empty((n_boot, m))
    for b in range(n_boot):
        if use_strata:
            idx = np.concatenate([g[rng.integers(len(g), size=len(g))]
                                  for g in groups])
        else:
            idx = rng.integers(n, size=n)
        S = D[idx]
        boot_mean[b] = S.mean(0)
        boot_med[b] = np.median(S, 0)
    lo, hi = 100 * alpha / 2, 100 * (1 - alpha / 2)
    return [{"metric": pd_.names[j],
             "mean_lo": float(np.percentile(boot_mean[:, j], lo)),
             "mean_hi": float(np.percentile(boot_mean[:, j], hi)),
             "median_lo": float(np.percentile(boot_med[:, j], lo)),
             "median_hi": float(np.percentile(boot_med[:, j], hi)),
             "stratified": bool(use_strata), "n_strata": len(groups),
             "n_boot": int(n_boot)}
            for j in range(m)]


# ==========================================================================
# Assemblage
# ==========================================================================
#: REGLE DE TRAITEMENT DES METRIQUES DEGENEREES, fixee avant le pilote.
#:
#: Une metrique dont les differences appariees ont une variance nulle ne
#: peut pas etre studentisee. Deux cas, traites differemment :
#:
#:   constant_zero     toutes les differences valent exactement 0. Il n'y a
#:                     aucun effet a mesurer : statistique 0, p = 1. La
#:                     metrique reste dans la famille.
#:   constant_nonzero  toutes les differences sont egales et non nulles.
#:                     C'est un effet parfaitement reproductible, pas une
#:                     absence d'information : la statistique vaut 1 et la
#:                     p-valeur atteint le plancher exact 2^(1-n).
#:
#: Dans les deux cas la metrique est **conservee dans la famille
#: confirmatoire**. La retirer parce que sa difference est nulle sur un
#: echantillon serait une selection fondee sur les donnees, et elle
#: modifierait la taille de famille donc les p-valeurs corrigees des autres
#: metriques. Une metrique a variance nulle contribue 0 au maximum familial
#: et ne penalise donc pas les autres.
#:
#: Le drapeau ``degenerate`` est rapporte sur chaque ligne de sortie.
DEGENERATE_POLICY = "keep_in_family_and_flag"


def analyse_contrast(A: np.ndarray, names: Sequence[str],
                     conditions: Sequence[str], family: Sequence[str],
                     a: str = "C4", b: str = "C3",
                     strata: Optional[np.ndarray] = None,
                     n_perm: int = 10000, n_boot: int = 5000,
                     seed: int = 0) -> Tuple[List[Dict], PairedData]:
    """Chaine complete pour un contraste et une famille pre-declaree.

    La famille est testee **en un seul appel** : une seule matrice de
    differences, une seule procedure de permutation, une seule correction
    max-statistic. Les memes exemples sont donc retenus pour toutes les
    metriques, et les exclusions sont communes.
    """
    idx = [list(names).index(f) for f in family if f in names]
    sel = [list(names)[i] for i in idx]
    pdata = paired_differences(A[:, :, idx], sel, conditions, a, b, strata)
    desc = {d["metric"]: d for d in describe(pdata)}
    test = {t["metric"]: t for t in sign_flip_test(pdata, n_perm, seed)}
    boot = {c["metric"]: c for c in bootstrap_ci(pdata, n_boot, seed=seed)}
    rows = []
    for name in sel:
        r = {"contrast": f"{a}-{b}"}
        r.update(desc[name])
        r.update({k: v for k, v in test[name].items() if k != "metric"})
        r.update({k: v for k, v in boot[name].items() if k != "metric"})
        r["n_dropped"] = int(pdata.dropped.size)
        rows.append(r)
    return rows, pdata
