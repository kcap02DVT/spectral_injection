# Rapport d'audit — `spectral_injection`

*Étape 1 du protocole. Aucun fichier du pipeline n'a été modifié pour produire ce rapport.*

**Date** : 9 novembre 2026
**Périmètre** : 29 modules, 11 471 lignes, plus `spectral-trust` 0.3.0.
**Méthode** : lecture du code, et vérification numérique sur le cache d'attention
existant (37 paires BIPIA `train`, 40 `test`, Qwen2.5-0.5B-Instruct) pour tous
les points marqués *mesuré*.

---

## 1. Architecture

Pas de structure en paquets : 29 modules à plat sous `spectral_injection/`.
Quatre strates, par ordre de sédimentation.

| strate | modules | rôle |
|---|---|---|
| noyau | `config`, `prompts`, `runner`, `graphs`, `metrics`, `viz` | prompt, capture, graphe, métriques, figures |
| protocole | `bipia`, `controls`, `evaluate`, `specificity`, `report`, `verdict` | données, contrôles appariés, statistiques |
| hypothèses | `hypotheses*`, `topology`, `gsp_hypotheses`, `cluster_purity`, `h10_*` | campagnes successives, un CLI chacune |
| récent | `sinks`, `lambda2_topo`, `wavelets`, `wavelet_zscore`, `shape_stats`, `attn_cache` | Kron, ondelettes, formes, cache |

Le point d'entrée canonique est `runner.AttentionRunner` ; tout en dépend.
`attn_cache` découple capture et analyse, ce qui rend toute nouvelle hypothèse
gratuite en passes de modèle — c'est la pièce la plus réutilisable du dépôt.

**Dette** : `h10_sweep.py` est remplacé par `h10_vec.py` (même balayage, ~20×
plus rapide) mais les deux coexistent. Six points d'entrée CLI aux conventions
divergentes.

## 2. Dépendances

`spectral-trust==0.3.0` épinglé, et c'est bien la dernière version publiée
(9 septembre 2026). L'épinglage est justifié : la 0.2.x utilisait la
normalisation `rw` par défaut, un vecteur de départ Lanczos non initialisé, et
filtrait les valeurs propres inférieures à 1e-6 — exactement le régime que
λ₂ est censé détecter.

### Capacités de la bibliothèque jamais utilisées

| disponible | pertinence |
|---|---|
| `DirectedTopologist` — `L = I − D⁻¹A` asymétrique, `max_imaginary` | H3 : le dépôt symétrise tout et jette la direction |
| `layer_delta_features(diagnostics, metric)` | H6 : différences inter-couches par tête |
| `calculate_spectral_velocity` | H6 |
| `eig_energy`, `eig_spectral_entropy`, `eig_hfer` (famille *attention*) | H4 |

Piège de nommage : `METRIC_FAMILIES` classe `hfer` comme *hybrid*, alors que
`PER_HEAD_METRIC_NAMES` contient un `hfer` calculé à partir des seules valeurs
propres — c'est en réalité l'`eig_hfer` du registre. Préciser lequel dans tout
écrit.

## 3. Pipeline de capture

```
RunConfig(model, dtype="float32")
  → AutoModelForCausalLM(attn_implementation="eager")
  → une passe avant, output_attentions=True
  → [a[0].detach().cpu().float().numpy() for a in out.attentions]
```

**Correct** : `eager` imposé avec erreur explicite si l'attention revient vide ;
pas de génération ; `max_tokens=512` lève plutôt que de tronquer ; vérification
`n_tok == len(labelled)` interdisant toute désynchronisation étiquettes/attention.

**Défaut mémoire** : `out.attentions` retient toutes les couches simultanément
sur le device avant rapatriement. Pic = `L × H × T² × 4 o`, soit 705 Mo pour
Llama-3.2-3B à T=512.

### GQA — vérifié

Le tenseur capturé porte le nombre de têtes **de requête** (post-`repeat_kv`).
Qwen2.5-0.5B : 14 Q / 2 KV. Llama-3.2-1B : 32 Q / 8 KV. Gemma-2-2b : 8 Q / 4 KV,
plus `sliding_window=4096` sans effet aux longueurs actuelles mais qui mordrait
au-delà.

*Mesuré* — les têtes d'un même groupe KV ne sont pas indépendantes :

```
corrélation moyenne de λ₂, têtes du même groupe KV   +0.247
                           groupes différents         +0.155
rang participatif du spectre de corrélation           5.6 sur 14 têtes
3 premières composantes                               64 % de la variance
```

**Portée de ce chiffre, et sa limite** : le rang participatif est un indicateur
heuristique de dimensionnalité effective. Il **ne constitue pas** un nombre de
tests indépendants et ne doit pas être substitué à une correction de
multiplicité. Le traitement correct reste celui qu'applique déjà le dépôt —
permutation sur la statistique maximale, qui préserve la dépendance entre
cellules et reste valide quelle que soit la structure de corrélation. Le chiffre
sert uniquement à interpréter : une cellule retenue n'est pas isolée, ses
voisines du même groupe KV portent largement la même information.

## 4. Conventions de graphe

| # | vérification | verdict |
|---|---|---|
| 1 | Laplacien combinatoire ou normalisé | **Normalisé `sym`** par défaut, `rw` disponible. Le combinatoire `D − W` n'apparaît que dans `sinks.remove_kron`, où il est requis. ✔ |
| 2 | Degrés après symétrisation | Oui, `D` calculé sur `W = ½(A+Aᵀ)`. ✔ |
| 3 | Prétraitements cohérents par tête / agrégé | **NON** — voir §5.1 ✘ |
| 4 | Valeurs propres | `eigh` dense, triées, clippées à 0. Solveur creux interdit avec justification chiffrée (`hfer` faux de 12 % à n=329). ✔ |
| 5 | Retrait du sink | Modifie massivement la géométrie — §5.2 ✘ |
| 6 | Dépendance au nombre de tokens | Non neutralisée — §5.3 ✘ |
| 7 | Architectures | GQA correct ; sliding window non testé ; aucune garde sur attention non standard. ⚠ |
| 8 | Réutilisabilité des métriques | Large — §7 ✔ |
| 9 | Nœuds de degré nul | `np.where(d > EPS, 1/√d, 0)` partout ; la bibliothèque laisse 1 sur la diagonale d'un nœud isolé. ✔ |
| 10 | Tests | `selftest.py` (8 blocs, sans modèle) + `--selftest` dans `wavelets` et `wavelet_zscore`. Pas de pytest, pas de CI. `sinks`, `lambda2_topo`, `h10_vec`, `shape_stats` sans tests. ⚠ |

## 5. Défauts confirmés par mesure

### 5.1 Les deux voies ne filtrent pas les mêmes nœuds

`metrics.per_head_layer_metrics` passe le tenseur **brut** à
`spectral_trust.per_head_metrics` : aucun filtrage. `graphs.build_graph` applique
`keep_mask` (template, BOS, puits) avant le Laplacien.

Les deux familles de résultats ne sont donc pas comparables : les AUROC de
0,83–0,92 par tête viennent de graphes contenant BOS et le template ; l'échec de
`cut_conductance` est mesuré sur des graphes filtrés. Toute phrase qui les oppose
est invalide.

`drop_sink` et `drop_template` valent `False` par défaut depuis le mode texte
brut, pour une raison correcte et documentée : sans BOS, le puits serait un token
*de contenu* trouvé par `argmax` indépendamment dans chaque condition, et les
deux graphes perdraient des nœuds différents.

### 5.2 Le retrait du puits détruit la connectivité qu'il fournissait

*Mesuré*, mêmes cellules, par tête :

```
retrait      λ₂ : A → B               alignement de u₂ (A vs B)
drop         0.1521 → 0.0422  (−72 %)         0.579
renorm       0.1521 → 0.0486  (−68 %)         0.527
kron         0.1995 → 0.2158  (+8 %)          0.964
```

Le puits porte 4–5 % de l'énergie de u₂ pour 21,5 % du volume : il n'est pas
*dans* le mode de Fiedler, il est ce qui tient le graphe connexe.

**Statut de la réduction de Kron** : c'est une **représentation alternative**,
pas la correction optimale du retrait du sink. Elle préserve les résistances
effectives entre nœuds gardés — propriété mathématique établie — mais cela ne
prouve pas qu'elle préserve ce qui compte pour la question posée. L'impact sur la
détection est d'ailleurs faible : AUROC appariée 0,786 (drop) / 0,820 (Kron) /
0,881 (sans retrait). Les trois représentations doivent être traitées comme trois
choix de modélisation à comparer, pas comme un correct et deux faux.

### 5.3 `sym` borne le spectre mais ne découple pas la longueur

*Mesuré* sur quatre conditions du même prompt JBB :

```
T = 68  →  λ₂ = 0.3795        T = 97  →  λ₂ = 0.3413
T = 74  →  λ₂ = 0.3654        T = 100 →  λ₂ = 0.3473
```

λ₂ décroît avec T sur les deux voies. Le spectre est borné dans [0,2]
indépendamment de T, mais λ₂ lui-même tombe quand le graphe grandit — comportement
attendu d'un graphe de diamètre croissant. **Borner n'est pas découpler.**

L'affirmation du README selon laquelle `sym` « neutralise le confond de longueur »
est trop forte et doit être corrigée. Elle est reprise telle quelle dans
`results/bipia/main.tex`.

## 6. Incompatibilités avec les contraintes de l'étude

- **« Aucun classificateur »** → `specificity.probe()` est une régression
  logistique ridge à 5 plis. Hors périmètre.
- **« Aucune sélection optimisée sur les labels »** → `specificity.select_cells()`,
  `transfer.py` et `shape_stats.evaluate()` sélectionnent la cellule maximisant
  l'effet. Sous la nouvelle contrainte, ces résultats sont **descriptifs**, pas
  prédictifs.

Le reste de l'appareil statistique est compatible : permutation par inversion de
signe préservant la dépendance, maximum familial, Benjamini-Hochberg, Holm,
Freedman-Lane pour les covariables de longueur. **Manque : le bootstrap**, absent
du dépôt.

## 7. Couverture des hypothèses H1–H6

| hyp. | réutilisable | manquant |
|---|---|---|
| H1 connectivité | `layer_metrics`, `per_head_layer_metrics`, `graphs.spectrum`, `topology` | distribution complète du spectre ; `eig_*` de la 0.3.0 |
| H2 frontières | `metrics.cut_statistics` (cut, ncut, conductance, modularité, densités par paires) | décomposition S / U / E ; cas dégénérés non testés |
| H3 flux | `specificity.attention_measures` sur l'attention brute orientée | matrice de flux inter-régions ; `DirectedTopologist` |
| H4 signaux | `wavelets.chebyshev_apply` accepte un signal arbitraire | **énergie de Dirichlet sur indicateur de région — absente** |
| H5 communautés | `cluster_purity`, `h10_vec` | stabilité des partitions entre couches |
| H6 trajectoires | λ₂ par couche dans `viz` | variation totale, ruptures de tendance ; `layer_delta_features` |

### Réserve sur H4

Pour un signal indicateur binaire `x = 1_E` et le Laplacien **combinatoire** :

```
xᵀ(D − W)x  =  Σ_{i∈E, j∉E} W_ij  =  cut(E, V∖E)
```

L'énergie de Dirichlet combinatoire est donc **exactement** la coupe pondérée de
H2 : elle n'apporte aucune information nouvelle. Toute valeur de H4 repose sur ce
qu'ajoutent les **normalisations** — Laplacien `sym`, normalisation du signal,
volume des régions — et cela doit être évalué explicitement plutôt que supposé.

## 8. Reproductibilité — insuffisante

`attn_cache` enregistre par paire : position, catégorie, tokens insérés, longueur,
caractères.

Ne sont enregistrés nulle part : révision du modèle, révision du dataset, versions
des bibliothèques, seed effectif, configuration complète, hash de commit. Six des
onze éléments requis manquent.

## 9. Recommandations avant l'étape suivante

1. Module de **provenance** sérialisant config, versions, révisions et seed à
   chaque run. Condition de tout le reste.
2. **Troisième rôle** dans `prompts.py` pour séparer S / U / E, avec tests de
   frontière. Distinguer *connaître les frontières de E* (légitime, vient de la
   construction) de *connaître le label d'attaque* (interdit).
3. **Réutiliser** `DirectedTopologist` et `layer_delta_features` plutôt que
   réécrire H3 et H6.
4. **Implémenter** l'énergie de Dirichlet normalisée (H4), après avoir établi ce
   qu'elle ajoute à la coupe.
5. **Corriger** l'affirmation sur le confond de longueur dans le README et
   `main.tex` ; traiter T comme covariable systématique.
6. Traiter les trois retraits de sink comme des **représentations alternatives**
   à comparer.
