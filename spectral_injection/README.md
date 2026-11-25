# spectral-injection

Analyse du graphe d'attention pour la détection d'injection de prompt.
Pour une paire (prompt hôte, instruction injectée), le kit produit :

- les **métriques spectrales** par couche, famille *attention* uniquement ;
- les **statistiques de coupe** entre tokens bénins et tokens injectés ;
- la **figure** graphe circulaire + matrice d'adjacence.

## Installation

```bash
pip install -r requirements.txt
```

`spectral-trust` est épinglé en **0.3.0**. Les versions 0.2.x ne sont pas
comparables : la normalisation par défaut est passée de `rw` à `sym`, le
démarrage Lanczos non initialisé a été corrigé, et un filtre sur les valeurs
propres `>1e-6` écartait exactement le régime λ₂ proche de zéro que la
connectivité algébrique est censée détecter.

## Vérification sans modèle

```bash
python selftest.py
```

Huit blocs : bicluster planté, contrôle uniforme, effet du sink, bornes des
métriques, rendu de la figure, étiquetage par segments sur un tokenizer
simulé, AUROC, et chargeur BIPIA. Si ce test échoue, le bug est dans
l'analyse, pas dans le modèle. Aucun GPU, aucun modèle — seul le bloc BIPIA
touche le réseau, et il s'ignore proprement s'il est coupé.

## Utilisation — BIPIA email QA

```bash
# 1. choisir la couche sur le split train (catégories d'attaque A)
python -m spectral_injection.bipia_cli --model tinyllama --split train --n-pairs 30

# 2. rapporter sur le split test (catégories d'attaque B, disjointes)
python -m spectral_injection.bipia_cli --model tinyllama --split test \
    --n-pairs 50 --layer 8 --out results/bipia
```

Les fichiers BIPIA sont téléchargés au premier appel dans `data/bipia/`
(email + text_attack uniquement ; Web QA et Summarization demandent les
scripts amont et ne sont pas utilisés).

Sorties dans `results/bipia/<modèle>/<split>/` :

| Fichier | Contenu |
|---|---|
| `per_pair_metrics.csv` | une ligne par (paire, condition, couche) |
| `summary_by_layer.csv` | AUROC et moyennes par couche |
| `figure_L*.png` | figure qualitative sur la première paire |

Et en console : AUROC par couche, ventilation par position d'insertion et par
catégorie d'attaque.

**Le protocole compte.** Les catégories d'attaque de `train` et `test` sont
disjointes (15 de chaque côté, 75 instructions chacune). Choisir la couche sur
`train` et la rapporter sur `test` est donc une évaluation hors distribution
par construction. Choisir la couche sur `test` et rapporter le même chiffre
est de la sélection sur le jeu d'évaluation — le CLI le signale.

## Utilisation — prompt unique

```bash
python -m spectral_injection.cli --example --model tinyllama

python -m spectral_injection.cli \
    --model llama3.2-1b \
    --benign-file host.txt \
    --injection-file attack.txt \
    --layer 8 --per-head --out results
```

Sorties dans `results/<modèle>/` : la figure, `metrics_benign.csv`,
`metrics_injected.csv`, la trajectoire de Fiedler par couche, et un résumé
console de la couche choisie.

Options utiles :

| Option | Effet |
|---|---|
| `--color-by cluster` | colore par clustering spectral 3-way (k-means sur u₂, u₃) au lieu de bleu/rouge |
| `--keep-sink` | garde le puits d'attention (il relie les deux familles et referme la coupe) |
| `--keep-template` | garde les tokens du système et du template |
| `--normalization rw` | reproduit les chiffres d'avant la 0.3.0 |
| `--per-head` | exporte les métriques par tête sur le span de l'injection |

## Ce que produit le pipeline

### Métriques spectrales (`layer_metrics`)

`fiedler_value`, `connectivity_ratio`, `spectral_entropy_norm`, `hfer`,
`spectral_radius` — mêmes noms, même ordre et mêmes définitions que
`spectral_trust.per_head.PER_HEAD_METRIC_NAMES`, donc directement comparables
avec la sortie par tête de la bibliothèque.

Toutes sont de famille **attention** : calculables à partir des poids
d'attention seuls. Les métriques `energy`, `smoothness_index`,
`spectral_entropy` et `hfer` au sens *hybride* de spectral-trust projettent le
flux résiduel sur la base propre de l'attention ; elles exigent l'accès aux
états cachés et ne sont volontairement pas utilisées ici.

### Statistiques de coupe (`cut_statistics`)

| Champ | Lecture |
|---|---|
| `density_benign`, `density_injection` | poids moyen **par paire possible** à l'intérieur de chaque famille |
| `density_cross` | idem entre les deux familles |
| `separation` | `sqrt(intra_b · intra_i) / cross`, > 1 si les familles se tiennent plus qu'elles ne se parlent |
| `conductance`, `normalized_cut`, `modularity` | qualité de la coupe entre bénin et injection |

La normalisation par le nombre de paires n'est pas cosmétique : l'injection
est presque toujours plus courte que l'hôte, et une somme brute de poids
favoriserait mécaniquement le groupe le plus grand.

### Coupe découverte (`fiedler_partition`, `partition_quality`)

Un détecteur ne sait pas où est l'injection. Il découvre une coupe par le
signe du vecteur de Fiedler et mesure sa qualité. `cut_conductance` se calcule
donc aussi sur un prompt bénin, où la meilleure coupe disponible reste
mauvaise — c'est ça, le signal. `partition_agreement` compare ensuite la coupe
découverte à la vraie frontière (`accuracy`, `ARI`) : c'est ce qui fait passer
de « le graphe se sépare en deux » à « le graphe se sépare le long de
l'injection ».

## Choix de conception

**Symétrisation.** L'attention d'un décodeur est triangulaire inférieure.
Sans `W = (A + Aᵀ)/2`, toute quantité spectrale non orientée est vide de sens.

**Laplacien `sym` par défaut.** Spectre dans [0, 2] quelle que soit la
longueur, donc un prompt bénin et sa variante injectée plus longue restent
comparables. C'est ce qui neutralise le confound de longueur — inutile de
diviser les métriques par `token_length` à la main.

**Retrait du puits.** Le premier token capte une part énorme de la masse
d'attention et relie tout le monde à tout le monde. Laissé en place, il
referme la coupe qu'on cherche à mesurer. `selftest.py` mesure cet effet.

**Retrait du template.** Le prompt système est identique dans les deux
conditions : c'est du bruit constant.

**Étiquetage exact des tokens.** Le composite est construit ici, donc la
frontière est connue ; les rôles viennent de l'`offset_mapping` du tokenizer
rapide. Aucun alignement approximatif entre deux tokenisations, et aucune
troncature silencieuse — `runner.analyse` lève une erreur si le nombre de
labels ne correspond pas au nombre de positions d'attention.

**Têtes.** `aggregate_heads` sert à la figure et aux trajectoires
qualitatives. Pour toute expérience de **détection**, utilisez `--per-head` :
moyenner les têtes dilue l'anomalie de routage, portée par quelques têtes,
dans un fond quasi uniforme.

## Limite à garder en tête pour la figure

Dans le panneau circulaire, les nœuds sont placés en ordre de lecture : une
injection contiguë occupe donc un arc contigu **par construction**. Le
regroupement visuel y est en partie un artefact du layout. La matrice
d'adjacence réordonnée par rôle et les statistiques de coupe sont la preuve
non ambiguë ; le cercle est là pour l'intuition.

## Structure

```
spectral_injection/
    config.py     RunConfig, alias de modèles
    prompts.py    construction du composite, étiquetage exact des tokens
    runner.py     chargement du modèle, capture de l'attention (eager)
    graphs.py     symétrisation, Laplaciens, Fiedler, filtrage des nœuds
    metrics.py    métriques spectrales, coupe, partition découverte
    viz.py        figure graphe circulaire + adjacence
    bipia.py      chargeur BIPIA email QA, insertion start/middle/end
    evaluate.py   boucle sur les paires, AUROC, agrégations
    cli.py        point d'entrée prompt unique
    bipia_cli.py  point d'entrée BIPIA
selftest.py       validation sans modèle
```
