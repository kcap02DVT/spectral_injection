"""Rapport markdown court, genere depuis les metadonnees et resultats."""

import json
from typing import Dict


def write_report(path: str, meta: Dict, results: Dict) -> str:
    L = []
    L.append("# Pilote par tête — rapport généré\n")
    L.append(f"Modèle : `{meta['model']['id']}` — {meta['model']['n_layers']} couches × "
             f"{meta['model']['n_heads']} têtes (kv={meta['model']['n_kv_heads']}), "
             f"device `{meta['model']['device']}`, float32, `attn_implementation=eager`.\n")
    L.append("## Format et système\n")
    L.append(f"Template : **officiel du modèle** (`apply_chat_template`), documenté dans le notebook.  ")
    ds = meta["template"]
    L.append(f"Le template injecte-t-il un système par défaut quand aucun n'est passé ? "
             f"**{'OUI' if ds['default_system_injected'] else 'non'}**"
             + (f" — contenu : `{ds['default_system_content']}`" if ds['default_system_injected'] else "") + ".\n")
    L.append(f"- xTRam1 : configuration primaire = système **{meta['xtram1']['primary_system']}** ; "
             f"sensibilité sur {', '.join(meta['xtram1']['system_configs'])}. "
             f"Le système est identique entre les deux conditions d'une même configuration.")
    L.append(f"- SPML : System Prompt **fourni, varié** ({meta['spml']['n_system_prompts_distinct']} distincts) ; "
             f"paires construites **au sein du même System Prompt** (identique dans la paire).\n")
    L.append("## Règle de sélection des têtes (déclarée avant les résultats)\n")
    L.append(f"Toutes les têtes. Test apparié par inversion de signe sur chaque cellule (couche, tête) ; "
             f"compte principal = Benjamini-Hochberg q={meta['stats']['fdr_q']} ; compte secondaire = "
             f"max-statistic FWER. Métrique primaire : `{meta['stats']['primary_metric']}`. "
             f"Cellule déclarée pour le vecteur de Fiedler et les adjacences : "
             f"couche {meta['declared_cell'][0]}, tête {meta['declared_cell'][1]}. "
             f"**Aucune moyenne de λ₂ entre têtes dans les résultats principaux.**\n")
    L.append("## Appariement\n")
    for k in ("xtram1", "spml"):
        p = meta[k]["pairing"]
        L.append(f"- **{k}** : {p['n_pairs']} paires ; écart résiduel en tokens — "
                 f"médian |gap| {p['gap_abs_median']}, max |gap| {p['gap_abs_max']} ; "
                 f"journal : {json.dumps({kk: vv for kk, vv in p.items() if kk.startswith(('adv_','ctl_','systems_','pairs_'))})}")
    L.append("")
    L.append("## Résultats principaux (par cellule)\n")
    for k, r in results.items():
        c = r["counts"]
        L.append(f"- **{k}** — {r['n_pairs']} paires, {c['n_cells']} cellules : "
                 f"BH q<.05 → **{c['sig_bh']}** ({c['sig_bh_plus']} +, {c['sig_bh_minus']} −) ; "
                 f"FWER → **{c['sig_fwer']}** ({c['sig_fwer_plus']} +, {c['sig_fwer_minus']} −).")
    L.append("")
    if "sensitivity" in meta:
        L.append("## Sensibilité au système (xTRam1)\n")
        L.append("Corrélation de Spearman des 336 Δλ₂ médians par cellule entre configurations :\n")
        for line in meta["sensitivity"].get("spearman", []):
            L.append(f"- {line}")
        L.append(f"\nCellules significatives (BH) par configuration : "
                 f"{json.dumps(meta['sensitivity'].get('sig_bh_by_config', {}))}\n")
    L.append("## Limites\n")
    for lim in meta["limits"]:
        L.append(f"- {lim}")
    L.append("")
    L.append("## Ce qui n'est PAS défendable\n")
    L.append("- Moyenner λ₂ entre têtes et présenter le résultat comme principal : noie un effet localisé.")
    L.append("- Choisir la « meilleure » cellule après avoir vu la carte couche × tête.")
    L.append("- Comparer adverse et contrôle sans appariement en longueur : λ₂ et les masses spectrales dépendent de T.")
    L.append("- Lire la classification comportementale automatique comme une mesure : elle est provisoire, l'annotation manuelle est en attente.")
    txt = "\n".join(L)
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(txt)
    return txt
