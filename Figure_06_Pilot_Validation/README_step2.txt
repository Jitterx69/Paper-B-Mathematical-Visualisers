Figure 6, Step 2: per-GoP analytic model versus measurement extraction

Project prerequisites:
- data/bbb352.yuv (exactly 43,253,760 bytes)
- data/eval_res_baseline.json copied from Figure_02_PreChannel_Distortion/eval_res.json
- scripts/evallab_fig2.py copied from Figure_02_PreChannel_Distortion/scripts/evallab_fig2.py
- scripts/figure6_per_gop.py and data/table_x_reference.csv from this package

Run from the Figure_06_Pilot_Validation project root:
  source ../Figure_01_Kalman_Allocation/.venv/bin/activate
  python -m py_compile scripts/figure6_per_gop.py
  python scripts/figure6_per_gop.py

Expected output:
- results/figure6_proposed_per_gop.csv (64 GoP records)
- results/figure6_per_gop_summary.json

The script replicates the Figure 2 pilot's validated 16-GoP recursion at 6, 8, 10, and 12 dB. It checks the per-GoP aggregates against the earlier eval_res.json and checks measured means/minima of the four schemes against Table X. It does not invent individual H.264 GoP values; the codec baseline is reported from the manuscript table only.
