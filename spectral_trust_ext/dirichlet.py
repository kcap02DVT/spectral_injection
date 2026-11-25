"""Energie de Dirichlet PAR TETE, sur le flux residuel.

Ce module est destine a vivre dans ``spectral_trust``. Il n'implemente
AUCUNE algebre de graphe lui-meme : la symetrisation et la construction du
laplacien passent par ``GraphConstructor``, afin que la convention de graphe
soit exactement celle du reste de la bibliotheque.

CE QUI CHANGE PAR RAPPORT A ``SpectralAnalyzer.compute_dirichlet_energy``.
La fonction de la bibliotheque recoit UN laplacien par couche, construit sur
l'attention deja moyennee entre les tetes par ``aggregate_heads``. Ici la
moyenne entre tetes n'est jamais prise avant le graphe : chaque tete donne
son propre W, son propre L, et sa propre energie.

    W_h = symmetrize_attention(A_h)          # GraphConstructor, par tete
    L_h = construct_laplacian(W_h)           # GraphConstructor, par tete
    E_h = tr(X^T L_h X)                      # la formule de la bibliotheque

La formule scalaire est inchangee -- c'est litteralement
``tr(X^T L X)`` de ``spectral.py``. Seul l'ordre des operations change.

POURQUOI CELA NE CHANGE RIEN POUR LE LAPLACIEN COMBINATOIRE. ``D(W) =
diag(W 1)`` est lineaire en W, donc ``L = D - W`` l'est aussi, et
``tr(X^T L X)`` egalement. Il s'ensuit l'identite EXACTE

    (1/H) sum_h tr(X^T L(W_h) X)  =  tr(X^T L((1/H) sum_h W_h) X)

pour ``normalization='none'``. Passer par les tetes y est un non-evenement,
et c'est pourquoi ce cas est conserve : il sert de CONTROLE D'INTEGRITE du
code par tete, au meme titre que ``h2_dirichlet_comb`` controle ``h2_cut``.

Pour ``normalization='sym'`` l'identite tombe, parce que ``D^(-1/2)`` est
applique avec les degres PROPRES a chaque tete. L'ecart entre les deux
ordres d'operation est alors une quantite mesuree, pas une erreur, et
``head_aggregation_gap`` la rend.

CE QUI SORT, de forme [couches, tetes] -- rien n'est moyenne a l'interieur
du module, l'agregation est laissee a l'appelant :

    energy      [L, H]   tr(X_l^T L_lh X_l)
    smoothness  [L, H]   energy / ||X_l||_F^2, le quotient de Rayleigh
                         matriciel ; borne par lambda_max(L) <= 2 pour
                         'sym', donc comparable entre couches et entre
                         prompts de longueurs differentes
    signal_norm [L]      ||X_l||_F^2, rendu separement parce que la norme
                         du flux residuel croit fortement avec la
                         profondeur : un Delta sur ``energy`` melange un
                         effet de graphe et un effet de norme, alors que
                         ``smoothness`` ne retient que le premier

AVERTISSEMENT DE CADRE. Le signal X est le flux residuel : ces quantites
sont HYBRIDES au sens de spectral-trust, pas attention-only. Elles ne
peuvent pas etre rapportees dans la meme famille que les metriques
calculees sur les seuls poids d'attention. ``FAMILY`` le declare et doit
etre propage jusqu'au rapport.
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence

import numpy as np
import torch

try:                                    # usage depuis l'interieur du paquet
    from .graph import GraphConstructor
except ImportError:                     # usage depuis le depot
    from spectral_trust.graph import GraphConstructor

__all__ = ["DirichletEnergy", "DirichletBlocks", "FAMILY",
           "per_head_laplacians", "dirichlet_energy",
           "dirichlet_energy_all_layers", "head_aggregation_gap",
           "dirichlet_blocks", "dirichlet_blocks_all_layers",
           "check_block_decomposition"]

#: Famille de tenseur au sens de spectral-trust. Le signal est le flux
#: residuel, donc ces metriques exigent les etats caches.
FAMILY = "hybrid"


@dataclass
class DirichletEnergy:
    """Sortie de ``dirichlet_energy_all_layers``.

    ``energy`` et ``smoothness`` sont [couches, tetes]. ``signal_norm`` est
    [couches]. ``head_mean`` et ``head_std`` sont des AGREGATS DECLARES,
    fournis par commodite ; la carte par cellule reste la sortie primaire.
    """

    energy: np.ndarray
    smoothness: np.ndarray
    signal_norm: np.ndarray
    normalization: str
    family: str = FAMILY

    @property
    def head_mean(self) -> Dict[str, np.ndarray]:
        """Moyenne sur les tetes, par couche. Agregation declaree."""
        return {"energy": self.energy.mean(axis=1),
                "smoothness": self.smoothness.mean(axis=1)}

    @property
    def head_std(self) -> Dict[str, np.ndarray]:
        """Dispersion entre tetes, par couche. A rapporter avec la moyenne :
        une moyenne dont l'ecart-type vaut le tiers n'est pas un resume."""
        return {"energy": self.energy.std(axis=1),
                "smoothness": self.smoothness.std(axis=1)}

    def to_dict(self) -> Dict:
        return {"energy": self.energy, "smoothness": self.smoothness,
                "signal_norm": self.signal_norm,
                "normalization": self.normalization, "family": self.family}


def _as_tensor(x, dtype=torch.float64) -> torch.Tensor:
    if isinstance(x, torch.Tensor):
        return x.detach().to(dtype)
    return torch.as_tensor(np.asarray(x), dtype=dtype)


def per_head_laplacians(constructor: GraphConstructor,
                        attention_layer) -> torch.Tensor:
    """``A`` [H, T, T] -> ``L`` [H, T, T], une tete = un graphe.

    Les H tetes sont passees a ``construct_laplacian`` comme une dimension
    de lot : la bibliotheque calcule les degres sur le dernier axe, donc
    chaque tete recoit ses propres degres. C'est exactement ce que la
    moyenne prealable des matrices detruit.
    """
    A = _as_tensor(attention_layer)
    if A.ndim != 3:
        raise ValueError(f"attention_layer doit etre [H, T, T], recu {tuple(A.shape)}")
    W = constructor.symmetrize_attention(A.unsqueeze(0))[0]      # [H, T, T]
    return constructor.construct_laplacian(W)                    # [H, T, T]


def dirichlet_energy(constructor: GraphConstructor, attention_layer,
                     signals) -> np.ndarray:
    """Energie de Dirichlet de ``signals`` sur le graphe de CHAQUE tete.

    ``signals`` est [T, d] -- les etats caches de la couche. Rend [H].

        E_h = tr(X^T L_h X) = sum_d x_d^T L_h x_d

    La trace somme sur les d canaux, sans moyenne : c'est la convention de
    ``SpectralAnalyzer.compute_dirichlet_energy``, conservee telle quelle.
    """
    L = per_head_laplacians(constructor, attention_layer)         # [H, T, T]
    X = _as_tensor(signals)
    if X.ndim == 1:
        X = X.unsqueeze(1)
    if X.ndim != 2:
        raise ValueError(f"signals doit etre [T, d], recu {tuple(X.shape)}")
    if X.shape[0] != L.shape[-1]:
        raise ValueError(f"signals a {X.shape[0]} tokens, le graphe en a "
                         f"{L.shape[-1]}")
    # tr(X^T L_h X) pour chaque tete, sans former les matrices [d, d]
    e = torch.einsum("td,hts,sd->h", X, L, X)
    return e.cpu().numpy()


def dirichlet_energy_all_layers(constructor: GraphConstructor,
                                attention: Sequence,
                                signals: Sequence,
                                layers: Optional[Sequence[int]] = None
                                ) -> DirichletEnergy:
    """Toutes les couches. ``attention[l]`` [H, T, T], ``signals[l]`` [T, d].

    ``signals`` doit deja etre aligne sur ``attention`` : si les etats
    caches viennent de ``output_hidden_states``, la couche ``l`` de
    l'attention correspond a ``hidden_states[l + 1]``, et c'est a
    l'appelant de faire ce decalage -- le module ne devine pas.
    """
    if len(attention) != len(signals):
        raise ValueError(f"{len(attention)} couches d'attention contre "
                         f"{len(signals)} de signaux ; le decalage "
                         f"hidden_states[l+1] est a la charge de l'appelant")
    idx = list(range(len(attention))) if layers is None else list(layers)
    E, S, Nrm = [], [], []
    for l in idx:
        X = _as_tensor(signals[l])
        if X.ndim == 1:
            X = X.unsqueeze(1)
        nrm = float(torch.sum(X * X))
        e = dirichlet_energy(constructor, attention[l], X)
        E.append(e)
        S.append(e / nrm if nrm > 1e-8 else np.zeros_like(e))
        Nrm.append(nrm)
    return DirichletEnergy(
        energy=np.stack(E), smoothness=np.stack(S),
        signal_norm=np.asarray(Nrm, dtype=np.float64),
        normalization=str(constructor.config.normalization))


def head_aggregation_gap(constructor: GraphConstructor, attention_layer,
                         signals) -> Dict[str, float]:
    """Ce que le passage par les tetes change, sur une couche.

    Compare la MOYENNE DES ENERGIES PAR TETE a l'ENERGIE DU GRAPHE MOYEN,
    c'est-a-dire l'ordre des operations qu'on veut remplacer.

    Pour ``normalization='none'`` les deux sont egales par linearite de
    ``L = D - W`` en ``W`` : ``rel`` doit valoir ~1e-16, et toute valeur
    plus grande signale un bug dans le chemin par tete.

    Pour ``normalization='sym'`` l'ecart est reel et c'est la quantite
    d'interet : il mesure ce que la moyenne prealable des matrices efface.
    """
    A = _as_tensor(attention_layer)
    X = _as_tensor(signals)
    if X.ndim == 1:
        X = X.unsqueeze(1)
    per_head = dirichlet_energy(constructor, A, X)
    W = constructor.symmetrize_attention(A.unsqueeze(0))[0]       # [H, T, T]
    W_bar = W.mean(dim=0, keepdim=True)                           # [1, T, T]
    L_bar = constructor.construct_laplacian(W_bar)[0]
    pooled = float(torch.einsum("td,ts,sd->", X, L_bar, X))
    mean_of = float(per_head.mean())
    denom = max(abs(pooled), 1e-30)
    return {"mean_of_energies": mean_of, "energy_of_mean_graph": pooled,
            "abs": abs(mean_of - pooled), "rel": abs(mean_of - pooled) / denom,
            "head_std": float(per_head.std()),
            "normalization": str(constructor.config.normalization)}


# --------------------------------------------------------------------------
# Decomposition en blocs regionaux
# --------------------------------------------------------------------------
#: Bloc declare comme principal AVANT tout calcul de contraste : c'est la
#: ou les deux instructions se rencontrent. ``task`` precede ``slot`` dans
#: le prompt, donc le masque causal annule A[task, slot] et ce bloc non
#: oriente vaut exactement la moitie du flux oriente slot -> task.
PRIMARY_BLOCK = "task--slot"


@dataclass
class DirichletBlocks:
    """Decomposition par bloc, [couches, tetes, K] pour les trois mesures.

    ``energy``   somme du bloc, divisee par N (comparable a ``region_energy``)
    ``density``  la meme somme divisee par le nombre de liens POSSIBLES du
                 bloc : comparable entre blocs de tailles differentes
    ``share``    part du bloc dans l'energie de Dirichlet totale. INVARIANTE
                 PAR CHANGEMENT D'ECHELLE du flux residuel : si X est
                 multiplie par c, numerateur et denominateur le sont par
                 c^2. C'est la lecture a privilegier, puisqu'elle ne peut
                 pas etre portee par la norme.
    ``total``    [couches, tetes] l'energie de Dirichlet complete, qui est
                 la somme exacte des K blocs
    """

    energy: np.ndarray
    density: np.ndarray
    share: np.ndarray
    total: np.ndarray
    block_names: List[str]
    normalization: str
    family: str = FAMILY

    def block(self, name: str, what: str = "share") -> np.ndarray:
        """Carte [couches, tetes] d'un bloc nomme."""
        k = self.block_names.index(name)
        return getattr(self, what)[:, :, k]


def _degree_scaling(W: torch.Tensor, normalization: str) -> torch.Tensor:
    """Facteur ``s_i`` tel que l'energie vaille 1/2 sum W_ij ||s_i x_i - s_j x_j||^2.

    'sym'  -> s_i = D_ii^(-1/2), degre nul donne s_i = 0, ce qui isole le
              noeud exactement comme le fait ``construct_laplacian``.
    'none' -> s_i = 1 (laplacien combinatoire).
    'rw'   -> refuse : L n'y est pas symetrique, la forme quadratique
              associee n'est pas une energie de Dirichlet.
    """
    if normalization == "none":
        return torch.ones(W.shape[:-1], dtype=W.dtype, device=W.device)
    if normalization != "sym":
        raise ValueError(f"normalization={normalization!r} : pas de forme de "
                         f"Dirichlet symetrique")
    deg = W.sum(dim=-1)
    s = torch.zeros_like(deg)
    m = deg > 1e-8
    s[m] = deg[m].clamp(min=1e-8).rsqrt()
    return s


def dirichlet_blocks(constructor: GraphConstructor, attention_layer, signals,
                     labels: Sequence[str], region_names: Sequence[str]):
    """Une couche -> (energy, density, share, total) de formes [H, K] / [H].

    Les blocs sont les K = R(R+1)/2 paires NON ORDONNEES de regions, dans
    l'ordre de ``block_names(region_names)`` -- le meme que
    ``region_energy``, pour que les deux decompositions se lisent cote a
    cote.
    """
    from .region_energy import block_pairs, region_indices

    A = _as_tensor(attention_layer)
    X = _as_tensor(signals)
    if X.ndim == 1:
        X = X.unsqueeze(1)
    norm = str(constructor.config.normalization)
    W = constructor.symmetrize_attention(A.unsqueeze(0))[0]      # [H, T, T]
    s = _degree_scaling(W, norm)                                 # [H, T]

    # ||s_i x_i - s_j x_j||^2 par tete, sans former [H, T, T, d]
    Xs = s.unsqueeze(-1) * X.unsqueeze(0)                        # [H, T, d]
    G = Xs @ Xs.transpose(-1, -2)                                # [H, T, T]
    dg = torch.diagonal(G, dim1=-2, dim2=-1)                     # [H, T]
    sq = dg.unsqueeze(-1) + dg.unsqueeze(-2) - 2.0 * G           # [H, T, T]
    P = W * sq                                                   # terme par paire
    N = P.shape[-1]

    idx = region_indices(labels, region_names)
    pairs = block_pairs(region_names)
    H = P.shape[0]
    E = torch.zeros((H, len(pairs)), dtype=P.dtype)
    poss = np.zeros(len(pairs))
    for k, (ja, jb) in enumerate(pairs):
        a, b = region_names[ja], region_names[jb]
        ia, ib = idx[a], idx[b]
        if ia.size == 0 or ib.size == 0:
            continue
        ta = torch.as_tensor(ia, device=P.device, dtype=torch.long)
        tb = torch.as_tensor(ib, device=P.device, dtype=torch.long)
        sub = P.index_select(-2, ta).index_select(-1, tb).sum((-2, -1))
        # le facteur 1/2 de la forme par paires : un bloc hors diagonale
        # est compte dans les deux sens par la symetrie de P, donc la
        # somme simple sur (r, s) donne deja le bon terme ; un bloc
        # diagonal est compte deux fois et doit etre halve.
        E[:, k] = 0.5 * sub if a == b else sub
        poss[k] = (ia.size * (ia.size - 1)) if a == b else (2 * ia.size * ib.size)

    tot = E.sum(dim=1)                                           # [H]
    e = (E / N).cpu().numpy()
    dens = np.divide(E.cpu().numpy(), poss[None, :],
                     out=np.zeros_like(E.cpu().numpy()), where=poss[None, :] > 0)
    t = tot.cpu().numpy()
    share = np.divide(E.cpu().numpy(), t[:, None],
                      out=np.zeros_like(E.cpu().numpy()),
                      where=np.abs(t[:, None]) > 1e-30)
    return e, dens, share, t


def dirichlet_blocks_all_layers(constructor: GraphConstructor,
                                attention: Sequence, signals: Sequence,
                                labels: Sequence[str],
                                region_names: Sequence[str]) -> DirichletBlocks:
    """Toutes les couches. ``signals`` doit deja etre aligne sur l'attention."""
    from .region_energy import block_names

    if len(attention) != len(signals):
        raise ValueError(f"{len(attention)} couches d'attention contre "
                         f"{len(signals)} de signaux")
    E, D, S, T = [], [], [], []
    for l in range(len(attention)):
        e, d, s, t = dirichlet_blocks(constructor, attention[l], signals[l],
                                      labels, region_names)
        E.append(e); D.append(d); S.append(s); T.append(t)
    return DirichletBlocks(
        energy=np.stack(E), density=np.stack(D), share=np.stack(S),
        total=np.stack(T), block_names=block_names(region_names),
        normalization=str(constructor.config.normalization))


def check_block_decomposition(constructor: GraphConstructor, attention_layer,
                              signals, labels: Sequence[str],
                              region_names: Sequence[str]) -> Dict[str, float]:
    """La somme des K blocs doit egaler tr(X^T L X) du laplacien de la
    bibliotheque.

    C'est le controle qui garantit que la forme par paires et le facteur
    d'echelle des degres sont corrects : il confronte un calcul fait a la
    main a ``construct_laplacian``, sans supposer leur accord. A appeler au
    moins une fois par campagne.
    """
    _, _, _, tot = dirichlet_blocks(constructor, attention_layer, signals,
                                    labels, region_names)
    ref = dirichlet_energy(constructor, attention_layer, signals)
    den = np.maximum(np.abs(ref), 1e-30)
    rel = np.abs(tot - ref) / den
    return {"rel_max": float(rel.max()), "rel_median": float(np.median(rel)),
            "normalization": str(constructor.config.normalization)}
