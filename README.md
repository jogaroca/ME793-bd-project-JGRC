# ME793-bd-project-JGRC

Repository for the ME793 Bd chemostat project (nonlinear observability + nonlinear estimator design).

## Quick start

From the repository root:

1) Install dependencies

```bash
pip install -r requirements_minimal.txt
```

2) (Recommended) Install the repo so the `Utility/` package is importable from anywhere

```bash
pip install -e .
```

3) Run scripts (examples)

```bash
python Phase_1_Bd_Dynamics_Demo/A_bd_dynamics.py
python Phase_2_Nonlinear_Observability/A_bd_empirical_observability.py
python Phase_3_Nonlinear_Estimation/A_bd_generate_synthetic_data.py
```

## Notes

- All reusable code (model, motifs, observability, EKF, plotting) lives in `Utility/`.
- Scripts under `Phase_*/` should be runnable without manual `sys.path` manipulation.
- Outputs are written under `results/` (ignored by git).
