"""H-A : ou vit le vecteur de Fiedler, et ce deplacement repond-il a l'injection ?

OBSERVABLES DECLAREES AVANT TOUTE EXECUTION. Toutes sont **region-libres** :
elles n'utilisent que des fenetres positionnelles, jamais une segmentation
systeme / utilisateur / injection.

    ipr_u2        sum(u2^4)           concentration de u2 (||u2|| = 1).
                                      1 = tout sur un token, 1/N = etale.
    argmax_rel    argmax|u2| / (N-1)  position relative du token dominant,
                                      dans [0, 1]. 1 = dernier token.
    tail_mass_5   sum_{i >= N-5} u2^2 masse sur les 5 dernieres positions.
    tail_mass_10  masse sur les 10 % derniers tokens (au moins 1).
    head_mass_5   masse sur les 5 PREMIERES positions (controle : le puits
                  d'attention principal y vit).

Hypothese H-A : sous injection, le maillon faible du graphe se deplace du
CONTENU vers la STRUCTURE. Predictions signees, posees avant mesure :

    argmax_rel   AUGMENTE  (le dominant migre vers la queue)
    tail_mass_5  AUGMENTE
    ipr_u2       indetermine  (concentre dans les deux cas)
    head_mass_5  indetermine  (controle)

``lambda_2`` est recalcule au passage et compare a ``fiedler_value`` de
``spectral-trust`` comme controle d'integrite.

Le signe de u2 est fixe (premiere composante non nulle positive) ; toutes
les observables ci-dessus sont de toute facon invariantes par u2 -> -u2.
"""

from typing import Dict, List, Sequence, Tuple

import numpy as np

#: Observables, dans l'ordre de declaration.
METRICS: Tuple[str, ...] = ("lambda_2", "ipr_u2", "argmax_rel",
                            "tail_mass_5", "tail_mass_10", "head_mass_5")

#: Predictions signees de H-A. None = aucune prediction.
PREDICTED_SIGN: Dict[str, str] = {
    "argmax_rel": "+", "tail_mass_5": "+",
    "lambda_2": None, "ipr_u2": None, "tail_mass_10": None, "head_mass_5": None,
}


def _laplacian_batch(A: np.ndarray) -> np.ndarray:
    """``A`` [H, T, T] -> ``L_sym`` [H, T, T], self-loops CONSERVEES.

    Memes conventions que ``GSPConfig(symmetrization='symmetric',
    normalization='sym', remove_self_loops=False)``. Le controle croise de
    ``lambda_2`` contre ``fiedler_value`` verifie cette equivalence.
    """
    A = np.asarray(A, dtype=np.float64)
    W = 0.5 * (A + np.swapaxes(A, -1, -2))
    d = W.sum(-1)
    inv = np.where(d > 1e-12, 1.0 / np.sqrt(np.maximum(d, 1e-12)), 0.0)
    L = -W * inv[..., :, None] * inv[..., None, :]
    idx = np.arange(W.shape[-1])
    L[..., idx, idx] += 1.0
    return 0.5 * (L + np.swapaxes(L, -1, -2))


def localization(A_layer: np.ndarray) -> Dict[str, np.ndarray]:
    """Une couche ``[H, T, T]`` -> dict d'observables ``[H]``.

    Une seule decomposition propre batchee pour toutes les tetes.
    """
    A = np.asarray(A_layer, dtype=np.float64)
    H, T, _ = A.shape
    vals, vecs = np.linalg.eigh(_laplacian_batch(A))      # [H,T], [H,T,T]
    u2 = vecs[:, :, 1]                                    # [H, T]
    m = u2 ** 2                                           # masse, somme = 1
    k10 = max(1, int(round(0.10 * T)))
    return {
        "lambda_2": np.clip(vals[:, 1], 0.0, 2.0),
        "ipr_u2": (m ** 2).sum(-1),
        "argmax_rel": np.argmax(m, axis=-1) / max(T - 1, 1),
        "tail_mass_5": m[:, max(T - 5, 0):].sum(-1),
        "tail_mass_10": m[:, T - k10:].sum(-1),
        "head_mass_5": m[:, :min(5, T)].sum(-1),
    }


def analyse_prompt(att: Sequence[np.ndarray]) -> Dict[str, np.ndarray]:
    """Toutes les couches -> dict d'observables ``[couches, tetes]``."""
    per_layer = [localization(A) for A in att]
    return {k: np.stack([d[k] for d in per_layer]) for k in METRICS}
