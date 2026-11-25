# Pilote par tête — rapport généré

Modèle : `Qwen/Qwen2.5-0.5B-Instruct` — 24 couches × 14 têtes (kv=2), device `cpu`, float32, `attn_implementation=eager`.

## Format et système

Template : **officiel du modèle** (`apply_chat_template`), documenté dans le notebook.  
Le template injecte-t-il un système par défaut quand aucun n'est passé ? **OUI** — contenu : `You are Qwen, created by Alibaba Cloud. You are a helpful assistant.`.

- xTRam1 : configuration primaire = système **vide** ; sensibilité sur vide, defaut_qwen, minimal. Le système est identique entre les deux conditions d'une même configuration.
- SPML : System Prompt **fourni, varié** (840 distincts) ; paires construites **au sein du même System Prompt** (identique dans la paire).

## Règle de sélection des têtes (déclarée avant les résultats)

Toutes les têtes. Test apparié par inversion de signe sur chaque cellule (couche, tête) ; compte principal = Benjamini-Hochberg q=0.05 ; compte secondaire = max-statistic FWER. Métrique primaire : `fiedler_value`. Cellule déclarée pour le vecteur de Fiedler et les adjacences : couche 12, tête 0. **Aucune moyenne de λ₂ entre têtes dans les résultats principaux.**

## Appariement

- **xtram1** : 20 paires ; écart résiduel en tokens — médian |gap| 0.0, max |gap| 0 ; journal : {"adv_too_long": 3, "adv_unmatched": 0, "ctl_pool": 1109, "ctl_pool_dropped_too_long": 91}
- **spml** : 20 paires ; écart résiduel en tokens — médian |gap| 13.0, max |gap| 31 ; journal : {"systems_considered": 116, "systems_unmatched": 83, "pairs_dropped_too_long": 13}

## Résultats principaux (par cellule)

- **xtram1_vide** — 20 paires, 336 cellules : BH q<.05 → **73** (68 +, 5 −) ; FWER → **15** (14 +, 1 −).
- **spml** — 20 paires, 336 cellules : BH q<.05 → **101** (27 +, 74 −) ; FWER → **21** (5 +, 16 −).
- **xtram1_defaut_qwen** — 20 paires, 336 cellules : BH q<.05 → **90** (84 +, 6 −) ; FWER → **28** (26 +, 2 −).
- **xtram1_minimal** — 20 paires, 336 cellules : BH q<.05 → **83** (74 +, 9 −) ; FWER → **22** (22 +, 0 −).

## Sensibilité au système (xTRam1)

Corrélation de Spearman des 336 Δλ₂ médians par cellule entre configurations :

- vide         vs defaut_qwen   rho=0.777  p=5.4e-69  meme signe sur 82.4% des cellules
- vide         vs minimal       rho=0.869  p=7.3e-104  meme signe sur 85.1% des cellules
- defaut_qwen  vs minimal       rho=0.812  p=3.1e-80  meme signe sur 86.0% des cellules

Cellules significatives (BH) par configuration : {"vide": 73, "defaut_qwen": 90, "minimal": 83}

## Limites

- Pilote : 20 + 20 paires, un seul modele (Qwen/Qwen2.5-0.5B-Instruct). Aucune generalisation.
- xTRam1 : controle ENTRE exemples (textes differents en tout sauf la longueur) ; les exemples adverses sont des phrases adversariales autonomes, pas des injections enfouies dans un contenu externe.
- SPML : systeme identique dans la paire, mais le User Prompt change de contenu, de forme et d'intention simultanement ; une difference par cellule n'isole pas la malveillance.
- Appariement en longueur : ecart residuel max 0 (xTRam1) et 31 (SPML) tokens ; 83 systemes SPML ecartes faute de controle assez proche.
- Aucune etiquette de succes d'attaque : la mesure comportementale est provisoire (regle de mots-cles) et non annotee.
- Aucune performance de detection n'est rapportee : un effet apparie n'est pas un detecteur.
- Analyse par Degree descriptive seulement (sous-puissance).
- per_head_metrics calcule en float32 : le controle croise lambda_2 porte un residu ~1e-7 attendu.

## Ce qui n'est PAS défendable

- Moyenner λ₂ entre têtes et présenter le résultat comme principal : noie un effet localisé.
- Choisir la « meilleure » cellule après avoir vu la carte couche × tête.
- Comparer adverse et contrôle sans appariement en longueur : λ₂ et les masses spectrales dépendent de T.
- Lire la classification comportementale automatique comme une mesure : elle est provisoire, l'annotation manuelle est en attente.