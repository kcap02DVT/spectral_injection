"""Tests du module statistique sur des donnees synthetiques a effet connu.

Aucun modele, aucune donnee reelle. Chaque cas a une reponse attendue posee
d'avance : effet nul, effet positif, effet negatif, donnees manquantes, et
dependance entre couches et tetes.

    python -m spectral_injection.opi.stats_selftest
"""

import numpy as np

from .stats import (PRIMARY_AGGREGATOR, aggregate_examples,
                    aggregate_trajectories, analyse_contrast, bootstrap_ci,
                    describe, paired_differences, sign_flip_test)

OK = []


def check(name, cond, detail=""):
    OK.append(bool(cond))
    print(f"  [{'ok ' if cond else 'ECHEC'}] {name}" + (f"   {detail}" if detail else ""))


def synth(n=40, L=6, H=8, effect=0.0, noise=1.0, seed=0, rho=0.0,
          missing=()):
    """Tenseur ``[n, 4, L, H, 2]``.

    ``effect`` est ajoute a la condition C4 (indice 3) sur la metrique 0.
    ``rho`` introduit une dependance : un decalage propre a l'exemple,
    partage par toutes ses couches et toutes ses tetes.
    """
    rng = np.random.default_rng(seed)
    V = rng.normal(0, noise, (n, 4, L, H, 2))
    if rho:
        V += rho * rng.normal(0, 1, (n, 1, 1, 1, 1))     # effet d'exemple
    V[:, 3, :, :, 0] += effect
    for i in missing:
        V[i, 2] = np.nan                                  # C3 manquante
    return V


CONDS = ["C1", "C2", "C3", "C4"]
NAMES = ["m_effet", "m_temoin"]


def main() -> int:
    print("1. Agregation : couches et tetes reduites, exemples conserves")
    V = synth(n=12, L=5, H=7)
    A = aggregate_examples(V)
    check("forme [n, cond, metriques]", A.shape == (12, 4, 2), f"{A.shape}")
    T = np.random.default_rng(0).normal(size=(12, 4, 7, 3))
    check("trajectoires [n, cond, stats]",
          aggregate_trajectories(T).shape == (12, 4, 3))
    man = np.nanmedian(V[3, 1], axis=(0, 1))
    check("mediane correcte sur un exemple",
          np.allclose(A[3, 1], man), f"ecart {np.abs(A[3,1]-man).max():.2e}")
    print()

    print("2. Effet nul : la p-valeur doit etre uniforme, pas petite")
    ps = []
    for s in range(30):
        A = aggregate_examples(synth(n=40, effect=0.0, seed=s))
        pdta = paired_differences(A, NAMES, CONDS)
        ps.append(sign_flip_test(pdta, n_perm=2000, seed=s)[0]["p_raw"])
    ps = np.array(ps)
    check("taux de rejet a 5 % proche de 5 %", (ps < 0.05).mean() <= 0.15,
          f"{(ps<0.05).mean():.3f} sur 30 repetitions")
    check("p-valeurs etalees sur [0,1]", ps.min() < 0.3 and ps.max() > 0.5,
          f"[{ps.min():.3f}, {ps.max():.3f}]")
    print()

    print("3. Effet positif : detecte, signe correct")
    A = aggregate_examples(synth(n=40, effect=0.8, noise=1.0, seed=1))
    pdta = paired_differences(A, NAMES, CONDS)
    d = describe(pdta)
    t = sign_flip_test(pdta, n_perm=5000, seed=1)
    check("moyenne de la difference positive", d[0]["mean"] > 0,
          f"{d[0]['mean']:+.3f}")
    check("dz eleve", d[0]["dz"] > 1.0, f"dz={d[0]['dz']:.2f}")
    check("p corrigee significative", t[0]["p_fwer"] < 0.01,
          f"p_fwer={t[0]['p_fwer']:.4f}")
    check("metrique temoin non significative", t[1]["p_fwer"] > 0.05,
          f"p_fwer={t[1]['p_fwer']:.4f}")
    print()

    print("4. Effet negatif : detecte, signe correct")
    A = aggregate_examples(synth(n=40, effect=-0.8, seed=2))
    pdta = paired_differences(A, NAMES, CONDS)
    d, t = describe(pdta), sign_flip_test(pdta, n_perm=5000, seed=2)
    check("moyenne negative", d[0]["mean"] < 0, f"{d[0]['mean']:+.3f}")
    check("win_rate sous 0.5", d[0]["win_rate"] < 0.5,
          f"{d[0]['win_rate']:.3f}")
    check("test bilateral : significatif malgre le signe",
          t[0]["p_fwer"] < 0.01, f"p_fwer={t[0]['p_fwer']:.4f}")
    print()

    print("5. Donnees manquantes : exemples ecartes, jamais imputes")
    V = synth(n=20, effect=0.8, seed=3, missing=(0, 5, 11))
    A = aggregate_examples(V)
    pdta = paired_differences(A, NAMES, CONDS)
    check("3 exemples ecartes", pdta.dropped.size == 3,
          f"{pdta.dropped.tolist()}")
    check("17 exemples conserves", pdta.n == 17, f"n={pdta.n}")
    check("indices ecartes corrects",
          set(pdta.dropped.tolist()) == {0, 5, 11})
    check("aucune valeur non finie dans les differences",
          np.isfinite(pdta.diff).all())
    print()

    print("6. Dependance couches/tetes : ne doit pas gonfler la puissance")
    #    Un decalage propre a l'exemple et partage par TOUTES ses conditions
    #    s'annule dans la difference appariee : il ne peut rien gonfler. La
    #    dependance qui compte est celle qui survit a l'appariement, c'est-a-
    #    dire une correlation entre cellules a l'interieur d'une seule
    #    condition. C'est ``cell_rho`` ci-dessous.
    #
    #    200 repetitions, et non 30 : a 30 l'ecart-type du taux estime vaut
    #    0,04, soit presque l'effet mesure. Un seuil pose sur 30 repetitions
    #    aurait echoue par bruit de Monte-Carlo, ce qui s'est produit.
    REP = 200
    se = (0.05 * 0.95 / REP) ** 0.5
    tol = 0.05 + 3 * se
    for label, cell_rho in (("cellules independantes", 0.0),
                            ("cellules correlees dans C4", 1.0)):
        ps = []
        for s in range(REP):
            rng = np.random.default_rng(5000 + s)
            V = rng.normal(0, 1, (30, 4, 6, 8, 2))
            if cell_rho:
                V[:, 3] += cell_rho * rng.normal(0, 1, (30, 1, 1, 2))
            A = aggregate_examples(V)
            pdta = paired_differences(A, NAMES, CONDS)
            ps.append(sign_flip_test(pdta, n_perm=999, seed=s)[0]["p_raw"])
        rate = float((np.array(ps) < 0.05).mean())
        check(f"{label} : taux de rejet calibre", rate <= tol,
              f"{rate:.3f}  (cible 0.05, tolerance {tol:.3f}, {REP} rep.)")
    print()

    print("7. Correction max-statistic : p_fwer >= p_raw toujours")
    rng = np.random.default_rng(7)
    V = rng.normal(size=(35, 4, 4, 6, 12))
    V[:, 3, :, :, 0] += 0.7
    A = aggregate_examples(V)
    nm = [f"m{j}" for j in range(12)]
    pdta = paired_differences(A, nm, CONDS)
    t = sign_flip_test(pdta, n_perm=4000, seed=7)
    check("p_fwer >= p_raw pour toutes les metriques",
          all(r["p_fwer"] >= r["p_raw"] - 1e-12 for r in t))
    n_raw = sum(r["p_raw"] < 0.05 for r in t)
    n_fw = sum(r["p_fwer"] < 0.05 for r in t)
    check("la correction reduit le nombre de rejets", n_fw <= n_raw,
          f"{n_raw} bruts -> {n_fw} corriges sur 12")
    print()

    print("8. Plancher bilateral : 2^(1-n), et non 2^-n")
    A = aggregate_examples(synth(n=5, effect=50.0, seed=8))
    pdta = paired_differences(A, NAMES, CONDS)
    t = sign_flip_test(pdta, n_perm=5000, seed=8)
    # La statistique est |.|, donc invariante par inversion globale : la
    # valeur extreme est atteinte par DEUX configurations, pas une.
    check("p_raw atteint exactement 2^(1-n)",
          abs(t[0]["p_raw"] - 2.0 ** -4) < 1e-12,
          f"p_raw={t[0]['p_raw']:.5f}, 2^(1-5)={2.0**-4:.5f}, "
          f"2^-5={2.0**-5:.5f}")
    check("plancher exact rapporte",
          abs(t[0]["p_floor_exact"] - 2.0 ** -4) < 1e-12)
    print()

    print("9. Bootstrap : couverture et stratification")
    A = aggregate_examples(synth(n=60, effect=0.5, seed=9))
    strata = np.array(["a"] * 20 + ["b"] * 20 + ["c"] * 20)
    pdta = paired_differences(A, NAMES, CONDS, strata=strata)
    ci = bootstrap_ci(pdta, n_boot=2000, seed=9)
    check("stratification active", ci[0]["stratified"] is True,
          f"{ci[0]['n_strata']} strates")
    check("IC contient l'effet vrai 0.5",
          ci[0]["mean_lo"] <= 0.5 <= ci[0]["mean_hi"],
          f"[{ci[0]['mean_lo']:.3f}, {ci[0]['mean_hi']:.3f}]")
    check("IC du temoin contient 0",
          ci[1]["mean_lo"] <= 0 <= ci[1]["mean_hi"],
          f"[{ci[1]['mean_lo']:.3f}, {ci[1]['mean_hi']:.3f}]")
    small = np.array(["a"] * 59 + ["z"])
    pd2 = paired_differences(A, NAMES, CONDS, strata=small)
    ci2 = bootstrap_ci(pd2, n_boot=500, seed=9)
    check("strate trop petite -> bootstrap simple, signale",
          ci2[0]["stratified"] is False)
    print()

    print("10. Chaine complete")
    A = aggregate_examples(synth(n=30, effect=0.6, seed=10))
    rows, pdta = analyse_contrast(A, NAMES, CONDS, NAMES, "C4", "C3",
                                  strata=np.array(["a"] * 15 + ["b"] * 15),
                                  n_perm=2000, n_boot=1000, seed=10)
    need = {"contrast", "metric", "n", "mean", "median", "dz", "win_rate",
            "p_raw", "p_fwer", "mean_lo", "mean_hi", "stratified", "n_dropped"}
    check("toutes les colonnes attendues", need <= set(rows[0]),
          f"manque {sorted(need - set(rows[0]))}")
    check("contraste etiquete", rows[0]["contrast"] == "C4-C3")
    print()

    print("11. Permutations EXACTES pour petits effectifs")
    A = aggregate_examples(synth(n=8, effect=3.0, seed=11))
    pdta = paired_differences(A, NAMES, CONDS)
    t = sign_flip_test(pdta, n_perm=5000, seed=11)
    check("methode exacte retenue a n=8", t[0]["method"] == "exact",
          f"{t[0]['method']}, {t[0]['n_perm_effective']} configurations")
    check("2^8 = 256 configurations enumerees",
          t[0]["n_perm_effective"] == 256)
    check("p_raw atteint le plancher exact 2^(1-n)",
          abs(t[0]["p_raw"] - 2.0 ** -7) < 1e-12,
          f"p_raw={t[0]['p_raw']:.6f}, plancher={2.0**-7:.6f}")
    check("p est un multiple exact de 1/256",
          abs(t[0]["p_raw"] * 256 - round(t[0]["p_raw"] * 256)) < 1e-9)
    # deux executions doivent donner exactement le meme resultat
    t2 = sign_flip_test(pdta, n_perm=5000, seed=999)
    check("resultat exact independant de la graine",
          t[0]["p_raw"] == t2[0]["p_raw"])
    big = aggregate_examples(synth(n=40, effect=0.5, seed=11))
    tb = sign_flip_test(paired_differences(big, NAMES, CONDS), n_perm=3000)
    check("Monte-Carlo au-dela du seuil", tb[0]["method"] == "monte_carlo")
    print()

    print("12. Planchers : bilateral exact vs resolution Monte-Carlo")
    for n_ex in (5, 8, 12):
        A = aggregate_examples(synth(n=n_ex, effect=5.0, seed=12))
        t = sign_flip_test(paired_differences(A, NAMES, CONDS))[0]
        check(f"n={n_ex} : plancher exact = 2^(1-n) = {2.0**(1-n_ex):.5f}",
              abs(t["p_floor_exact"] - 2.0 ** (1 - n_ex)) < 1e-12,
              f"rapporte {t['p_floor_exact']:.5f}")
    A = aggregate_examples(synth(n=40, effect=5.0, seed=12))
    t = sign_flip_test(paired_differences(A, NAMES, CONDS), n_perm=999)[0]
    check("Monte-Carlo : plancher = 1/(B+1)",
          abs(t["p_floor_mc"] - 1 / 1000) < 1e-12, f"{t['p_floor_mc']:.5f}")
    check("plancher applicable = max des deux",
          abs(t["p_floor"] - max(2.0 ** -39, 1 / 1000)) < 1e-12)
    print()

    print("13. Metriques degenerees : variance nulle")
    rng = np.random.default_rng(13)
    V = rng.normal(size=(10, 4, 3, 4, 3))
    V[:, 3, :, :, 0] = V[:, 2, :, :, 0]            # difference exactement 0
    V[:, 3, :, :, 1] = V[:, 2, :, :, 1] + 7.0      # difference constante +7
    A = aggregate_examples(V)
    nm = ["const_zero", "const_nonzero", "normale"]
    pdta = paired_differences(A, nm, CONDS)
    t = sign_flip_test(pdta)
    d = describe(pdta)
    check("difference nulle detectee", t[0]["degenerate"] == "constant_zero",
          f"{t[0]['degenerate']}")
    check("difference nulle -> p = 1", abs(t[0]["p_raw"] - 1.0) < 1e-12,
          f"p={t[0]['p_raw']:.4f}")
    check("difference constante non nulle detectee",
          t[1]["degenerate"] == "constant_nonzero", f"{t[1]['degenerate']}")
    check("effet parfait -> p au plancher exact, PAS p = 1",
          abs(t[1]["p_raw"] - 2.0 ** -9) < 1e-12,
          f"p={t[1]['p_raw']:.6f} (l'ancienne version rendait 1.0)")
    check("mediane de la difference constante = +7",
          abs(d[1]["median"] - 7.0) < 1e-9)
    check("les 3 metriques restent dans la famille", len(t) == 3)
    print()

    print("14. Famille unique de 12 : une seule correction")
    rng = np.random.default_rng(14)
    V = rng.normal(size=(30, 4, 4, 5, 12))
    V[:, 3, :, :, 0] += 1.0                         # une seule vraie
    A = aggregate_examples(V)
    nm12 = [f"m{j}" for j in range(12)]
    pdta = paired_differences(A, nm12, CONDS)
    t = sign_flip_test(pdta, n_perm=5000, seed=14)
    check("12 metriques testees ensemble", len(t) == 12)
    check("memes exemples pour toute la famille",
          all(r["n_perm_effective"] == t[0]["n_perm_effective"] for r in t))
    check("p_fwer >= p_raw partout",
          all(r["p_fwer"] >= r["p_raw"] - 1e-12 for r in t))
    check("la vraie metrique survit a la correction", t[0]["p_fwer"] < 0.05,
          f"p_fwer={t[0]['p_fwer']:.4f}")
    # correction commune contre deux familles separees : la commune doit
    # etre plus severe sur la metrique de tete
    pa = paired_differences(A[:, :, :10], nm12[:10], CONDS)
    ta = sign_flip_test(pa, n_perm=5000, seed=14)
    check("famille de 12 au moins aussi severe que famille de 10",
          t[0]["p_fwer"] >= ta[0]["p_fwer"] - 1e-9,
          f"12 -> {t[0]['p_fwer']:.4f}, 10 -> {ta[0]['p_fwer']:.4f}")
    print()

    n_ok, n = sum(OK), len(OK)
    print(f"{n_ok} / {n} tests passes")
    return 0 if n_ok == n else 1


if __name__ == "__main__":
    raise SystemExit(main())
