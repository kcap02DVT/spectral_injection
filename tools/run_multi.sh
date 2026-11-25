#!/usr/bin/env bash
set -u
cd /c/Users/damsw/Desktop/spectral_injection
run () {
  mid="$1"; slug="$2"
  echo "=========== $slug  $(date +%H:%M:%S)"
  python -m spectral_injection.opi.run --model "$mid" --per-stratum 4 --seed 0 \
      --out "results/opi/multi/$slug" > "results/opi/multi/$slug/run.log" 2>&1
  echo "=========== $slug termine rc=$? $(date +%H:%M:%S)"
}
run "Qwen/Qwen2.5-1.5B-Instruct"       qwen2.5-1.5b
run "google/gemma-2-2b-it"             gemma-2-2b
run "meta-llama/Llama-3.2-3B-Instruct" llama3.2-3b
echo "TOUT TERMINE $(date +%H:%M:%S)"
