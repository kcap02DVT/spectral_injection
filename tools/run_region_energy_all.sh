#!/usr/bin/env bash
# Campagne complete, modeles traites SEQUENTIELLEMENT.
#
# Un processus par modele : la memoire du modele precedent est rendue au
# systeme avant le chargement du suivant. Sur cette machine la RAM
# disponible (~10 GiB) ne permet pas deux modeles simultanement.
#
# L'ordre va du moins couteux au plus couteux, de sorte qu'un arret
# prematurer laisse quand meme plusieurs modeles exploitables.
#
# Un echec sur un modele n'interrompt pas la campagne : il est consigne et
# le modele suivant demarre. Le rapport final indique quels modeles ont
# abouti.
set -u

cd "$(dirname "$0")/.." || exit 1
OUT="results/opi/region_energy"
mkdir -p "$OUT"
LOG="$OUT/campaign.log"
: > "$LOG"

PER_STRATUM="${PER_STRATUM:-2}"
SEED="${SEED:-2027}"

run() {
  local tag="$1" model="$2"
  echo "=== $(date '+%H:%M:%S')  $tag  ($model)" | tee -a "$LOG"
  if python tools/run_region_energy.py \
        --model "$model" --tag "$tag" \
        --per-stratum "$PER_STRATUM" --seed "$SEED" \
        --asr --save-raw 2 --out "$OUT" >> "$LOG" 2>&1; then
    echo "    OK  $tag" | tee -a "$LOG"
  else
    echo "    ECHEC  $tag  (voir $OUT/$tag/run.log et $LOG)" | tee -a "$LOG"
  fi
}

run qwen2.5-0.5b   Qwen/Qwen2.5-0.5B-Instruct
run llama3.2-1b    meta-llama/Llama-3.2-1B-Instruct
run gemma-2-2b     google/gemma-2-2b-it
run qwen2.5-1.5b   Qwen/Qwen2.5-1.5B-Instruct
run llama3.2-3b    meta-llama/Llama-3.2-3B-Instruct
run phi-4-mini     microsoft/Phi-4-mini-instruct

echo "=== $(date '+%H:%M:%S')  campagne terminee" | tee -a "$LOG"
for d in "$OUT"/*/; do
  t=$(basename "$d")
  if [ -f "$d/provenance.json" ]; then
    python - "$d/provenance.json" <<'PY'
import json, sys
p = json.load(open(sys.argv[1], encoding="utf-8"))
print(f"  {p['tag']:16s} {p['n_layers']:3d}x{p['n_heads']:<3d} "
      f"paires {p['n_pairs_kept']:3d}/{p['n_pairs_planned']:<3d} "
      f"residu {float(p['identity_residual']):.1e}  {p['elapsed_s']:.0f}s")
PY
  else
    echo "  $t : pas de provenance (echec)"
  fi
done | tee -a "$LOG"
