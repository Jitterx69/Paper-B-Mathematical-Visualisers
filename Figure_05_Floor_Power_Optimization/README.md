# Figure 5 — Floor-Power Optimization, Controller and Mismatch

## Scope
Three quantitative groups based on Section VII and Tables V–VI of the manuscript:
- (a) Reported Table V floor-use fraction and equivalent throughput versus floor SNR.
- (b1) Equation (24) marginal-cost balance. The left side is evaluated exactly; the right side is a shape-preserving interpolation of the discrete dual values reported in Table V.
- (b2) Controller trajectory using the manuscript-reported local slope s approximately -0.52 and fixed point 9.41 dB. This is a local affine illustration, not a recomputation of the full nonlinear Gamma map.
- (c) Table VI mismatch regret results and Proposition 12 bound.

The optional scene-cut time-series inset is omitted because the manuscript reports aggregate factors but not a complete reproducible time-series input. The figure plan explicitly warns against showing a scene-cut time series unless it is actually simulated.

## Run
Requires Python 3, NumPy, SciPy and Matplotlib.

```bash
python -m py_compile scripts/figure5_floor_controller.py
python scripts/figure5_floor_controller.py
```

All outputs are stored in `results/`. Source tables and derived dense-curve/controller data are stored in `data/`.
