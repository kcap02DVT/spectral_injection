"""Tests unitaires H1-H4 sur des graphes dont les valeurs sont calculables.

Aucun modele, aucun reseau. Si ce fichier echoue, le bug est dans les
mathematiques, pas dans l'experience.

    python -m spectral_injection.opi.selftest
"""

import sys

import numpy as np

from .structural import (BANDS, BAND_NAMES, dirichlet_combinatorial, degrees,
                         h1_connectivity, h2_boundary, h3_flows,
                         h4_spectral_signal, laplacian_sym, spectrum,
                         symmetrize)

OK = []


def check(name, cond, detail=""):
    OK.append(bool(cond))
    print(f"  [{'ok ' if cond else 'ECHEC'}] {name}" + (f"   {detail}" if detail else ""))


def clique(n, w=1.0):
    W = np.full((n, n), w)
    np.fill_diagonal(W, 0.0)
    return W


def main() -> int:
    print("1. Graphe complet K_n : spectre analytique de L_sym")
    # L_sym(K_n) a pour valeurs propres 0 (x1) et n/(n-1) (x n-1)
    for n in (4, 7, 12):
        W = clique(n)
        vals, _ = spectrum(laplacian_sym(W))
        expected = np.r_[0.0, np.full(n - 1, n / (n - 1))]
        check(f"K_{n} spectre", np.allclose(vals, expected, atol=1e-10),
              f"max |ecart| = {np.abs(vals - expected).max():.2e}")
    print()

    print("2. Bornes et symetrie")
    rng = np.random.default_rng(0)
    A = np.tril(rng.random((40, 40)))
    A = A / A.sum(-1, keepdims=True)                # lignes stochastiques
    W = symmetrize(A)
    L = laplacian_sym(W)
    vals, vecs = spectrum(L)
    check("W symetrique", np.allclose(W, W.T, atol=1e-12))
    check("diagonale de W nulle", np.abs(np.diag(W)).max() == 0.0)
    check("L_sym symetrique", np.allclose(L, L.T, atol=1e-12))
    check("valeurs propres reelles et dans [0,2]",
          vals.min() >= -1e-12 and vals.max() <= 2 + 1e-9,
          f"[{vals.min():.3e}, {vals.max():.6f}]")
    check("lambda_1 = 0 sur graphe connexe", abs(vals[0]) < 1e-9,
          f"{vals[0]:.2e}")
    check("vecteurs propres orthonormes",
          np.allclose(vecs.T @ vecs, np.eye(len(vals)), atol=1e-9))
    print()

    print("3. Composantes connexes : multiplicite de la valeur propre nulle")
    for k in (2, 3):
        blocks = [clique(6) for _ in range(k)]
        Wb = np.zeros((6 * k, 6 * k))
        for i, b in enumerate(blocks):
            Wb[6 * i:6 * i + 6, 6 * i:6 * i + 6] = b
        v, _ = spectrum(laplacian_sym(Wb))
        h1 = h1_connectivity(v)
        check(f"{k} composantes -> n_zero = {k}", int(h1["n_zero"]) == k,
              f"n_zero={int(h1['n_zero'])}, lambda_2={h1['lambda_2']:.2e}, "
              f"premiere non nulle={h1['lambda_first_nonzero']:.4f}")
    print()

    print("4. Coupe, volumes, conductance, Ncut sur une bipartition connue")
    # deux cliques de poids 1 reliees par une seule arete de poids w0
    n1 = n2 = 5
    w0 = 0.3
    Wc = np.zeros((n1 + n2, n1 + n2))
    Wc[:n1, :n1] = clique(n1)
    Wc[n1:, n1:] = clique(n2)
    Wc[n1 - 1, n1] = Wc[n1, n1 - 1] = w0
    e = np.zeros(n1 + n2, bool)
    e[:n1] = True
    h2 = h2_boundary(Wc, e)
    vol_e = n1 * (n1 - 1) * 1.0 + w0
    check("cut exact", abs(h2["cut"] - w0) < 1e-12, f"{h2['cut']:.6f} vs {w0}")
    check("vol(E) exact", abs(h2["vol_E"] - vol_e) < 1e-12,
          f"{h2['vol_E']:.6f} vs {vol_e}")
    check("conductance exacte",
          abs(h2["conductance"] - w0 / min(vol_e, vol_e)) < 1e-12)
    check("Ncut exact",
          abs(h2["ncut"] - w0 * (1 / vol_e + 1 / vol_e)) < 1e-12)
    print()

    print("5. Controle mathematique : Dirichlet combinatoire = coupe")
    for seed in (0, 1, 2):
        r = np.random.default_rng(seed)
        Aw = np.tril(r.random((30, 30)))
        Aw = Aw / Aw.sum(-1, keepdims=True)
        Ww = symmetrize(Aw)
        m = r.random(30) < 0.3
        if m.all() or not m.any():
            continue
        d = dirichlet_combinatorial(Ww, m.astype(float))
        c = h2_boundary(Ww, m)["cut"]
        check(f"x^T(D-W)x = cut  (seed {seed})", abs(d - c) < 1e-10,
              f"{d:.10f} vs {c:.10f}")
    print()

    print("6. Controle mathematique : Rayleigh de D^(1/2)1_E = cut / vol(E)")
    for seed in (0, 1):
        r = np.random.default_rng(seed)
        Aw = np.tril(r.random((30, 30)))
        Aw = Aw / Aw.sum(-1, keepdims=True)
        Ww = symmetrize(Aw)
        m = r.random(30) < 0.35
        if m.all() or not m.any():
            continue
        dg = degrees(Ww)
        y = np.sqrt(dg) * m
        Ls = laplacian_sym(Ww)
        ray = float(y @ Ls @ y) / float(y @ y)
        h2b = h2_boundary(Ww, m)
        target = h2b["cut"] / h2b["vol_E"]
        check(f"Rayleigh = cut/vol(E)  (seed {seed})", abs(ray - target) < 1e-9,
              f"{ray:.10f} vs {target:.10f}")
    print()

    print("7. H4 : masses de bande, centroide, variante B")
    r = np.random.default_rng(3)
    Aw = np.tril(r.random((36, 36)))
    Aw = Aw / Aw.sum(-1, keepdims=True)
    Ww = symmetrize(Aw)
    v4, u4 = spectrum(laplacian_sym(Ww))
    m = np.zeros(36, bool)
    m[20:33] = True
    for var in ("A", "B"):
        h4 = h4_spectral_signal(v4, u4, m, var)
        s = sum(h4[b] for b in BAND_NAMES)
        check(f"variante {var} : masses de bande somment a 1",
              abs(s - 1.0) < 1e-9, f"{s:.12f}")
        check(f"variante {var} : centroide = quotient de Rayleigh",
              abs(h4["centroid"] - h4["rayleigh"]) < 1e-9,
              f"{h4['centroid']:.10f} vs {h4['rayleigh']:.10f}")
        check(f"variante {var} : HFER dans [0,1]", 0 <= h4["hfer"] <= 1,
              f"{h4['hfer']:.4f}")
    hA = h4_spectral_signal(v4, u4, m, "A")
    hB = h4_spectral_signal(v4, u4, m, "B")
    check("variante B retire bien la composante de valeur propre nulle",
          hB["band_0_05"] < hA["band_0_05"],
          f"A={hA['band_0_05']:.4f} -> B={hB['band_0_05']:.4f}")
    # B n'est pas une simple soustraction de moyenne : on le verifie
    xm = m.astype(float) - m.mean()
    xm /= np.linalg.norm(xm)
    pm = (u4.T @ xm) ** 2
    cent_mean = float((pm * v4).sum())
    check("variante B differe du centrage par la moyenne arithmetique",
          abs(hB["centroid"] - cent_mean) > 1e-6,
          f"B={hB['centroid']:.6f}, centrage naif={cent_mean:.6f}")
    print()

    print("8. H3 : orientation des flux preservee")
    T = 12
    Ad = np.zeros((T, T))
    Ad[8, 2] = 0.7            # un token de E regarde un token de U
    Ad[9, 9] = 0.1
    reg = np.array(list("TTUUUUTTEEEE"))
    f = h3_flows(Ad, reg)
    check("F_{E->U} capte l'arete descendante", abs(f["F_E_to_U"] - 0.7) < 1e-12,
          f"{f['F_E_to_U']:.3f}")
    check("F_{U->E} nul (causalite)", abs(f.get("F_U_to_E", 0.0)) < 1e-12)
    check("normalisation par |source|",
          abs(f["Fbar_E_to_U"] - 0.7 / 4) < 1e-12, f"{f['Fbar_E_to_U']:.5f}")
    print()

    n_ok, n = sum(OK), len(OK)
    print(f"{n_ok} / {n} tests passes")
    return 0 if n_ok == n else 1


if __name__ == "__main__":
    raise SystemExit(main())
