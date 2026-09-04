"""Regenerate both paper figures from the saved raw results.

Usage:
    python make_figures.py                 # reads ./data, writes ./figures
    python make_figures.py DATA_DIR OUT_DIR

Writes each figure as both PDF (for the manuscript) and PNG (for the README,
since GitHub Markdown cannot embed PDFs).
"""
import json
import os
import sys

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

DATA_DIR = sys.argv[1] if len(sys.argv) > 1 else 'data'
OUT_DIR = sys.argv[2] if len(sys.argv) > 2 else 'figures'
os.makedirs(OUT_DIR, exist_ok=True)

# ---------------------------------------------------------------- load
curv = np.load(os.path.join(DATA_DIR, 'raw_curvature.npz'))
spec = np.load(os.path.join(DATA_DIR, 'raw_spectral.npz'))
meta = json.load(open(os.path.join(DATA_DIR, 'meta.json')))

T = meta['T']
kappas = curv['kappas']
x = curv['sqkD'] if 'sqkD' in curv else np.array(
    [np.sqrt(k) * meta['D_DIAM'] for k in kappas])
reg_med = curv['reg_median']        # (n_kappa, T)
reg_fre = curv['reg_frechet']       # (n_kappa, T)

C_med = reg_med[:, -1] / np.sqrt(T)
C_fre = reg_fre[:, -1] / np.log(T)
zeta = x * np.cosh(x) / np.sinh(x)

print(f"curvature levels sqrt(kappa)*D = {np.round(x, 1).tolist()}")
print(f"  h-convex constant          {C_med[0]:.3f} -> {C_med[-1]:.3f}"
      f"  (x{C_med[-1] / C_med[0]:.2f})")
print(f"  strongly h-convex constant {C_fre[0]:.3f} -> {C_fre[-1]:.3f}"
      f"  (x{C_fre[-1] / C_fre[0]:.2f})")
print(f"  g-convex factor zeta       {zeta[0]:.2f} -> {zeta[-1]:.2f}"
      f"  (x{zeta[-1] / zeta[0]:.2f})")

plt.rcParams.update({'font.size': 11, 'axes.labelsize': 12,
                     'legend.fontsize': 9, 'axes.titlesize': 12})
FIGSIZE = (9.5, 3.6)


def save(fig, stem):
    for ext, kw in (('pdf', {}), ('png', {'dpi': 200})):
        path = os.path.join(OUT_DIR, f'{stem}.{ext}')
        fig.savefig(path, bbox_inches='tight', **kw)
        print(f"wrote {path}")
    plt.close(fig)


# ------------------------------------------- Figure 1: regret against t
tt = np.arange(1, T + 1)
cmap = plt.cm.viridis(np.linspace(0, 1, len(kappas)))
fig, ax = plt.subplots(1, 2, figsize=FIGSIZE)
for j, curves in enumerate([reg_med, reg_fre]):
    for c, xv, row in zip(cmap, x, curves):
        ax[j].plot(tt, row, color=c, lw=1.6, label=f'{xv:.0f}')
    ax[j].set_title('(a)' if j == 0 else '(b)', loc='left')
    ax[j].set_xlabel('round $t$')
    ax[j].set_ylabel('cumulative regret')
    ax[j].legend(title=r'$\sqrt{\kappa}D$')
    ax[j].grid(alpha=.3)
fig.tight_layout()
save(fig, 'fig_regret_vs_T')

# ------------------- Figure 2: curvature dependence and spectral gap
fig, ax = plt.subplots(1, 2, figsize=FIGSIZE)

ref = zeta / zeta[0] * C_med[0]      # scaled to start at the leftmost h-convex point
ax[0].plot(x, C_med, 'o-', label='D-ROGD, h-convex (median)')
ax[0].plot(x, C_fre, 's-', label='D-ROGD, strongly h-convex (Fréchet)')
ax[0].plot(x, ref, 'k--',
           label=r'g-convex factor $\sqrt{\kappa}D\coth(\sqrt{\kappa}D)$')
ax[0].set_xlabel(r'$\sqrt{\kappa}\,D$')
ax[0].set_ylabel('regret constant')
ax[0].set_title('(a)', loc='left')
ax[0].legend()
ax[0].grid(alpha=.3)

sig = spec['sigma2']
order = np.argsort(sig)
for key, mk, lb in [('reg_k1', 'o', r'$\kappa$=1'),
                    ('reg_k16', 's', r'$\kappa$=16')]:
    ax[1].plot(sig[order], spec[key][order], mk + '-', label=lb)
ax[1].set_xlim(-0.05, 1.0)
ax[1].set_xlabel(r'$\sigma_2(W)$   (worse connectivity $\rightarrow$)')
ax[1].set_ylabel(rf'cumulative regret at $T={T}$')
ax[1].set_title('(b)', loc='left')
ax[1].legend()
ax[1].grid(alpha=.3)

fig.tight_layout()
save(fig, 'fig_curv_net')
