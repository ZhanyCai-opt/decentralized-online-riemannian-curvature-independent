# Curvature-Independent Regret Bounds for Distributed Online Optimization on Hadamard Manifolds

Code and data for the numerical experiments on "Curvature-Independent Regret Bounds for Distributed Online Optimization on Hadamard Manifolds". **D-ROGD** (decentralized Riemannian online gradient
descent) is evaluated on a real concept hierarchy embedded in hyperbolic space, whose
curvature can be varied while the intrinsic scale of the data is held fixed.

## Layout

```
d_rogd_experiments.ipynb   full pipeline: data, embedding, experiments, figures
make_figures.py            redraw both figures from data/ without rerunning
figures/                   figures used in the paper (PDF and PNG)
data/                      raw arrays behind every figure
  raw_curvature.npz          cumulative regret curves, shape (n_kappa, T)
  raw_spectral.npz           final regret per topology, two curvatures
  embedding_U.npy            hyperbolic embedding of the concept hierarchy
  cluster_labels.npy         agent assignment of concepts
  meta.json                  hyperparameters
```

## Running

```bash
pip install -r requirements.txt
jupyter notebook d_rogd_experiments.ipynb     # ~30-60 min, GPU optional
python make_figures.py                        # seconds, uses data/ only
```

WordNet is fetched automatically through NLTK. The online loop runs on CPU; the GPU only
accelerates the embedding step.

## Setup in brief

The sub-hierarchy of the WordNet noun taxonomy rooted at `mammal.n.01` (300 synsets,
reduced to a tree) is embedded into a 10-dimensional Poincare ball by stress
minimisation. For each curvature `-kappa` the embedding is rescaled radially so the
intrinsic geodesic diameter is `D = 4`, making `sqrt(kappa) D` the only quantity that
varies. Eight agents each hold one k-means cluster of concepts, so local objectives are
heterogeneous. Objectives are the geometric median `d(x, Z)` (h-convex,
`eta_t = 1/sqrt(t)`) and the Frechet mean `0.5 d(x, Z)^2` (1-strongly h-convex,
`eta_t = 1/(mu t)`), over `T = 2000` rounds averaged across seeds 100-102. See the paper
for the full description.

## Notes on the implementation

Three choices remove confounds that would otherwise contaminate the curvature
comparison:

- **Perturbations are intrinsic.** The conformal factor `2/(1 - c|x|^2)` grows with both
  radius and curvature, so raw coordinate noise would make observation noise grow with
  curvature. `perturb()` divides it out, giving a fixed geodesic magnitude everywhere.
- **The consensus sub-problem is solved to a tolerance.** Its Karcher iteration
  contracts at a curvature-dependent rate, so a fixed step budget would be less accurate
  at higher curvature. An ablation cell confirms the results sit on a plateau.
- **The comparator is constrained to the feasible set.** An unconstrained solver drifts
  towards the ball boundary, inflating its loss and making regret spuriously negative;
  the notebook warns if this occurs.

## Curvature range

Holding `D` fixed while raising curvature pushes the data towards the ball boundary. In
float64 the target `D = 4` cannot be attained beyond about `sqrt(kappa) D = 16`: at
`kappa = 25, 36` the attained diameters are only 3.36 and 2.80. Those levels are
excluded from all reported results, and the notebook's feasibility check is run over the
wider set so the exclusion is explicit. Extending the range would need a better
conditioned model, such as the Lorentz (hyperboloid) model.

## License

MIT, see `LICENSE`.
