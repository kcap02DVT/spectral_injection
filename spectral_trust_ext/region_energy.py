"""Energie d'attention par regions, PAR COUCHE ET PAR TETE.

Ce module est destine a vivre dans ``spectral_trust``. Il n'implemente
AUCUNE algebre de graphe lui-meme : la symetrisation et la normalisation
passent par ``GraphConstructor``, afin que la convention de graphe soit
exactement celle du reste de la bibliotheque.

    W = symmetrize_attention(A)                  # GraphConstructor
    L = construct_laplacian(W)                   # GraphConstructor
    M = I - L                                    # = D^-1/2 W D^-1/2 quand
                                                 #   normalization == 'sym'
    Q = M * M,  Q_ii = 0                         # energie hors diagonale

``M = I - L`` est une identite, pas une reimplementation : pour
``normalization='sym'`` la bibliotheque pose L = I - D^-1/2 W D^-1/2, donc
I - L rend exactement l'adjacence normalisee. Le module leve si la
normalisation configuree n'est pas 'sym', car l'identite ne tiendrait plus.

QUANTITES RENDUES, toutes de forme [couches, tetes, ...] -- rien n'est
moyenne a l'interieur du module :

    energy      [L, H, K]   somme de Q sur le bloc, divisee par N
    density     [L, H, K]   la meme somme divisee par le nombre de liens
                            POSSIBLES du bloc : comparable entre blocs de
                            tailles differentes
    share       [L, H, K]   part du bloc dans l'energie hors diagonale
                            totale ; somme a 1 sur K par construction
    flow        [L, H, R, R] masse d'attention DIRIGEE brute (A, non
                            symetrisee) de la region requete i vers la
                            region cle j, divisee par le nombre de tokens
                            requete : lisible comme la part de l'attention
                            emise par i qui atterrit sur j
    offdiag     [L, H]      energie hors diagonale totale / N
    loop        [L, H]      variance des boucles propres M_ii
    variance    [L, H]      offdiag + loop

``variance`` est la variance du spectre de L. L'identite

    var(spec(L)) = [ sum_{i != j} M_ij^2 + sum_i (M_ii - mean M)^2 ] / N

est verifiee par ``check_variance_identity``, qui doit etre appelee au
moins une fois par campagne : elle controle d'un coup la symetrisation, la
normalisation et la mise a zero de la diagonale.

Les blocs sont les K = R(R+1)/2 paires non ordonnees de regions. Un bloc
(r, s) avec r != s compte les liens dans LES DEUX SENS, puisque Q est
symetrique ; le nombre de liens possibles vaut donc 2|r||s|, et |r|(|r|-1)
pour r == s. Une region vide donne zero partout, jamais NaN.
"""

from dataclasses import dataclass
from typing import Dict, List, Sequence, Tuple

import numpy as np
import torch

try:                                    # usage depuis l'interieur du paquet
    from .graph import GraphConstructor
except ImportError:                     # usage depuis le depot
    from spectral_trust.graph import GraphConstructor


__all__ = ["RegionEnergy", "block_pairs", "block_names", "region_indices",
           "normalized_adjacency", "region_energy",
           "region_energy_all_layers", "check_variance_identity"]


# --------------------------------------------------------------------------
# Nomenclature des blocs
# --------------------------------------------------------------------------
def block_pairs(region_names: Sequence[str]) -> List[Tuple[int, int]]:
    """Indices (i, j), i <= j, dans l'ordre de declaration des regions."""
    R = len(region_names)
    return [(i, j) for i in range(R) for j in range(i, R)]


def block_names(region_names: Sequence[str]) -> List[str]:
    """``['task--task', 'task--normal', ...]``, meme ordre que ``block_pairs``."""
    return [f"{region_names[i]}--{region_names[j]}"
            for i, j in block_pairs(region_names)]


def region_indices(labels: Sequence[str], region_names: Sequence[str]
                   ) -> Dict[str, np.ndarray]:
    """Etiquettes de tokens -> indices par region.

    Leve si une etiquette n'est pas declaree : une region silencieusement
    ignoree deplacerait toutes les energies sans erreur visible.
    """
    lab = np.asarray([str(x) for x in labels], dtype=object)
    unknown = sorted(set(lab.tolist()) - set(region_names))
    if unknown:
        raise ValueError(f"etiquettes non declarees : {unknown} ; "
                         f"regions attendues {list(region_names)}")
    return {r: np.flatnonzero(lab == r) for r in region_names}


# --------------------------------------------------------------------------
# Resultat
# --------------------------------------------------------------------------
@dataclass
class RegionEnergy:
    """Toutes les quantites, par couche et par tete. Rien d'agrege."""

    region_names: List[str]
    block_names: List[str]
    counts: Dict[str, int]
    energy: np.ndarray        # [L, H, K]
    density: np.ndarray       # [L, H, K]
    share: np.ndarray         # [L, H, K]
    flow: np.ndarray          # [L, H, R, R]
    offdiag: np.ndarray       # [L, H]
    loop: np.ndarray          # [L, H]
    variance: np.ndarray      # [L, H]
    n_tokens: int

    @property
    def n_layers(self) -> int:
        return int(self.energy.shape[0])

    @property
    def n_heads(self) -> int:
        return int(self.energy.shape[1])

    def block(self, name: str) -> np.ndarray:
        """Une carte [couches, tetes] pour un bloc nomme."""
        return self.energy[:, :, self.block_names.index(name)]

    def directed(self, query: str, key: str) -> np.ndarray:
        """Une carte [couches, tetes] pour un flux dirige nomme."""
        return self.flow[:, :, self.region_names.index(query),
                         self.region_names.index(key)]

    def to_dict(self) -> Dict:
        return {"region_names": list(self.region_names),
                "block_names": list(self.block_names),
                "counts": dict(self.counts),
                "n_layers": self.n_layers, "n_heads": self.n_heads,
                "n_tokens": int(self.n_tokens)}


# --------------------------------------------------------------------------
# Adjacence normalisee, via la bibliotheque
# --------------------------------------------------------------------------
def normalized_adjacency(constructor: GraphConstructor,
                         attention: torch.Tensor) -> torch.Tensor:
    """``M = I - L``, ou L est le Laplacien de ``GraphConstructor``.

    ``attention`` est [H, N, N] (une couche, toutes les tetes). La
    symetrisation et la normalisation sont celles de la configuration
    passee au constructeur ; seule ``normalization='sym'`` autorise
    l'identite M = I - L, d'ou le controle.
    """
    if constructor.config.normalization != "sym":
        raise ValueError(
            "region_energy exige normalization='sym' : l'identite M = I - L "
            f"ne tient pas pour {constructor.config.normalization!r}")
    W = constructor.symmetrize_attention(attention)
    L = constructor.construct_laplacian(W)
    N = L.shape[-1]
    eye = torch.eye(N, device=L.device, dtype=L.dtype)
    return eye - L


# --------------------------------------------------------------------------
# Une couche
# --------------------------------------------------------------------------
def _layer_quantities(constructor: GraphConstructor, A: torch.Tensor,
                      idx: Dict[str, np.ndarray],
                      region_names: Sequence[str]
                      ) -> Tuple[np.ndarray, ...]:
    """``A`` [H, N, N] -> (energy, density, flow, offdiag, loop) par tete."""
    M = normalized_adjacency(constructor, A).to(torch.float64)
    H, N, _ = M.shape
    diag = torch.diagonal(M, dim1=-2, dim2=-1)                  # [H, N]
    Q = M * M
    ii = torch.arange(N, device=M.device)
    Q[:, ii, ii] = 0.0

    loop = ((diag - diag.mean(dim=-1, keepdim=True)) ** 2).sum(-1) / N
    offdiag = Q.sum(dim=(-2, -1)) / N

    pairs = block_pairs(region_names)
    energy = torch.zeros((H, len(pairs)), dtype=torch.float64, device=M.device)
    density = torch.zeros_like(energy)
    for k, (i, j) in enumerate(pairs):
        x = idx[region_names[i]]
        y = idx[region_names[j]]
        if x.size == 0 or y.size == 0:
            continue
        tx = torch.as_tensor(x, device=M.device, dtype=torch.long)
        ty = torch.as_tensor(y, device=M.device, dtype=torch.long)
        z = Q[:, tx][:, :, ty].sum(dim=(-2, -1))
        if i == j:
            possible = x.size * (x.size - 1)
        else:
            z = z + Q[:, ty][:, :, tx].sum(dim=(-2, -1))
            possible = 2 * x.size * y.size
        energy[:, k] = z / N
        density[:, k] = z / max(possible, 1)

    R = len(region_names)
    Af = A.to(torch.float64)
    flow = torch.zeros((H, R, R), dtype=torch.float64, device=Af.device)
    for i, r in enumerate(region_names):
        x = idx[r]
        if x.size == 0:
            continue
        tx = torch.as_tensor(x, device=Af.device, dtype=torch.long)
        rows = Af[:, tx]                                        # [H, |r|, N]
        for j, s in enumerate(region_names):
            y = idx[s]
            if y.size == 0:
                continue
            ty = torch.as_tensor(y, device=Af.device, dtype=torch.long)
            flow[:, i, j] = rows[:, :, ty].sum(dim=(-2, -1)) / x.size

    return (energy.cpu().numpy(), density.cpu().numpy(), flow.cpu().numpy(),
            offdiag.cpu().numpy(), loop.cpu().numpy())


def region_energy(constructor: GraphConstructor, attention_layer,
                  labels: Sequence[str], region_names: Sequence[str]
                  ) -> RegionEnergy:
    """Une seule couche. ``attention_layer`` est [H, N, N]."""
    return region_energy_all_layers(constructor, [attention_layer],
                                    labels, region_names)


def region_energy_all_layers(constructor: GraphConstructor,
                             attentions: Sequence,
                             labels: Sequence[str],
                             region_names: Sequence[str]) -> RegionEnergy:
    """Toutes les couches. ``attentions`` est une sequence de [H, N, N].

    Le nombre de couches L et de tetes H est deduit de l'entree, jamais
    suppose : ``L = len(attentions)`` et ``H = attentions[0].shape[0]``.
    """
    if len(attentions) == 0:
        raise ValueError("aucune couche fournie")
    idx = region_indices(labels, region_names)
    N = len(labels)

    E, D, F, O, LO = [], [], [], [], []
    for A0 in attentions:
        A = torch.as_tensor(np.asarray(A0), dtype=torch.float64)
        if A.ndim != 3:
            raise ValueError(f"couche de forme {tuple(A.shape)}, "
                             f"attendu [H, N, N]")
        if A.shape[-1] != N or A.shape[-2] != N:
            raise ValueError(f"couche {tuple(A.shape)} incompatible avec "
                             f"{N} etiquettes de tokens")
        e, d, f, o, l = _layer_quantities(constructor, A, idx, region_names)
        E.append(e)
        D.append(d)
        F.append(f)
        O.append(o)
        LO.append(l)

    energy = np.stack(E)                                        # [L, H, K]
    offdiag = np.stack(O)                                       # [L, H]
    tot = energy.sum(-1, keepdims=True)
    share = np.divide(energy, tot, out=np.zeros_like(energy), where=tot > 0)
    loop = np.stack(LO)
    return RegionEnergy(
        region_names=list(region_names),
        block_names=block_names(region_names),
        counts={r: int(idx[r].size) for r in region_names},
        energy=energy, density=np.stack(D), share=share, flow=np.stack(F),
        offdiag=offdiag, loop=loop, variance=offdiag + loop, n_tokens=N)


# --------------------------------------------------------------------------
# Controle d'integrite
# --------------------------------------------------------------------------
def check_variance_identity(constructor: GraphConstructor, attention_layer,
                            head: int = 0) -> float:
    """Residu de ``var(spec(L)) == (offdiag + loop)``, sur une tete.

    Controle d'un coup la symetrisation, la normalisation et la mise a zero
    de la diagonale. Un residu superieur a ~1e-10 signale une divergence de
    convention entre ce module et ``GraphConstructor``.
    """
    A = torch.as_tensor(np.asarray(attention_layer), dtype=torch.float64)
    one = A[head:head + 1]
    M = normalized_adjacency(constructor, one).to(torch.float64)[0]
    N = M.shape[-1]
    L = torch.eye(N, dtype=torch.float64) - M
    L = 0.5 * (L + L.T)
    spec = torch.linalg.eigvalsh(L)
    diag = torch.diagonal(M)
    off = M.clone()
    off.fill_diagonal_(0.0)
    rhs = ((off ** 2).sum() + ((diag - diag.mean()) ** 2).sum()) / N
    return float(abs(torch.var(spec, unbiased=False) - rhs))
