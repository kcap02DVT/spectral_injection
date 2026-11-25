"""H-A : localisation du vecteur de Fiedler, tous prompts, toutes cellules."""
import os, sys, time
import numpy as np, pandas as pd
sys.path.insert(0, ".")
from spectral_injection.perhead import capture, localization as loc

OUT = "results/perhead/HA"
os.makedirs(OUT, exist_ok=True)
keys = sorted(k[:-4] for k in os.listdir("results/perhead/attention") if k.endswith(".npz"))
rows, t0 = [], time.time()
for i, key in enumerate(keys):
    ds, cfg, pid, cond = key.split("__")
    att = capture.load_attention(f"results/perhead/attention/{key}.npz")
    out = loc.analyse_prompt(att)
    nL, nH = out["lambda_2"].shape
    for m in loc.METRICS:
        v = out[m]
        for l in range(nL):
            for h in range(nH):
                rows.append((ds, cfg, int(pid[1:]), cond, l, h, m, float(v[l, h])))
    if (i + 1) % 20 == 0:
        print(f"  [{i+1}/{len(keys)}] {time.time()-t0:.0f}s", flush=True)
df = pd.DataFrame(rows, columns=["dataset", "system_config", "exemple_id",
                                 "condition", "couche", "tete", "metrique", "valeur"])
df.to_parquet(f"{OUT}/localization_long.parquet", index=False)
print(f"\n{len(df):,} lignes -> {OUT}/localization_long.parquet  ({time.time()-t0:.0f}s)")
