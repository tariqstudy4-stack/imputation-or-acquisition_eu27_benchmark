"""Regenerate every figure in ../figures/ from the CSVs in ../results/.

Run after experiment_final_eu.py (all three patterns) and gsd_sensitivity.py have
produced results/results_final_eu.csv and results/gsd_sensitivity_eu.csv.

    cd code
    python make_figures.py
"""
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.join(HERE, '..', 'results')
FIGURES = os.path.join(HERE, '..', 'figures')
os.makedirs(FIGURES, exist_ok=True)

METHOD_STYLE = {
    'PEDIGREE':   dict(color='#555555', label='Pedigree proxy (current practice)'),
    'KNN':        dict(color='#1b9e56', label='KNN donor imputation'),
    'SOFTIMPUTE': dict(color='#8c56a8', label='SoftImpute (low-rank)'),
    'VAE':        dict(color='#e08214', label='Variational autoencoder'),
    'GNN':        dict(color='#d7301f', label='Graph neural network'),
}
ERROR_METHODS = ['PEDIGREE', 'KNN', 'SOFTIMPUTE', 'VAE', 'GNN']
COVERAGE_METHODS = ['PEDIGREE', 'KNN', 'VAE', 'GNN']  # SOFTIMPUTE has no uncertainty estimate

PATTERNS = [
    ('MCAR',  'Random loss (MCAR)',                    'Random loss'),
    ('FLOW',  'Sensor failure (flow-structured)',       'Sensor failure'),
    ('BLOCK', 'Supplier non-disclosure (column blocks)', 'Supplier non-disclosure'),
]

full = pd.read_csv(os.path.join(RESULTS, 'results_final_eu.csv'))
summ = full.groupby(['method', 'pattern', 'rate'])[['mdape', 'flips', 'coverage']].mean().reset_index()
summ.to_csv(os.path.join(RESULTS, 'results_final_eu_summary.csv'), index=False)
gsd = pd.read_csv(os.path.join(RESULTS, 'gsd_sensitivity_eu.csv'))

RATES_PCT = [5, 10, 20, 40]


def _series(df, method, pattern, col):
    d = df[(df.method == method) & (df.pattern == pattern)].sort_values('rate')
    return d[col].values * 100.0


def plot_error(ax, pattern, title):
    for m in ERROR_METHODS:
        y = _series(summ, m, pattern, 'mdape')
        ax.plot(RATES_PCT, y, 'o-', color=METHOD_STYLE[m]['color'], label=METHOD_STYLE[m]['label'], linewidth=2, markersize=7)
    ax.axhline(100, color='gray', linestyle=':', linewidth=1)
    ax.text(RATES_PCT[0], 100, '100% error', color='gray', fontsize=8, va='bottom')
    ax.set_yscale('log')
    ax.set_xlabel('Missingness rate (%)')
    ax.set_ylabel('Impact-space median error, MdAPE (%, log scale)')
    ax.set_title(f'Point-estimate error — {title}')
    ax.grid(True, which='both', alpha=0.3)


def plot_coverage(ax, pattern, title):
    for m in COVERAGE_METHODS:
        y = _series(summ, m, pattern, 'coverage')
        ax.plot(RATES_PCT, y, 'o-', color=METHOD_STYLE[m]['color'], label=METHOD_STYLE[m]['label'], linewidth=2, markersize=7)
    ax.axhline(95, color='gray', linestyle='--', linewidth=1.5)
    ax.text(RATES_PCT[0], 96, 'nominal 95%', color='gray', fontsize=9)
    ax.set_ylim(0, 100)
    ax.set_xlabel('Missingness rate (%)')
    ax.set_ylabel('Coverage of 95% intervals (%)')
    ax.set_title(f'Uncertainty calibration — {title}')
    ax.grid(True, alpha=0.3)


# ---- individual per-pattern charts ----
for pat, title, _ in PATTERNS:
    fig, ax = plt.subplots(figsize=(7, 5))
    plot_error(ax, pat, title)
    ax.legend(loc='lower right', fontsize=9)
    fig.tight_layout()
    fig.savefig(os.path.join(FIGURES, f'chart_error_{pat}.png'), dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 5))
    plot_coverage(ax, pat, title)
    ax.legend(loc='best', fontsize=9)
    fig.tight_layout()
    fig.savefig(os.path.join(FIGURES, f'chart_coverage_{pat}.png'), dpi=150)
    plt.close(fig)

# ---- ranking-flip chart (pedigree baseline, all three patterns) ----
fig, ax = plt.subplots(figsize=(7, 5.5))
flip_colors = {'MCAR': 'tab:blue', 'FLOW': 'tab:orange', 'BLOCK': 'tab:green'}
flip_labels = {'MCAR': 'Random loss (MCAR)', 'FLOW': 'Sensor failure (flow-structured)', 'BLOCK': 'Supplier non-disclosure (column blocks)'}
for pat, _, _ in PATTERNS:
    y = _series(summ, 'PEDIGREE', pat, 'flips')
    ax.plot(RATES_PCT, y, 'o-', color=flip_colors[pat], label=flip_labels[pat], linewidth=2, markersize=7)
ax.set_xlabel('Missingness rate (%)')
ax.set_ylabel('Material ranking flips (%)')
ax.set_title('Decision stability under current practice (pedigree proxy)')
ax.legend(loc='upper left', fontsize=9)
ax.grid(True, alpha=0.3)
fig.tight_layout()
fig.savefig(os.path.join(FIGURES, 'chart_flips_pedigree.png'), dpi=150)
plt.close(fig)

# ---- combined 2x3 grid ----
fig, axes = plt.subplots(2, 3, figsize=(20, 10.7))
fig.suptitle('EU-27 benchmark: pedigree, KNN, low-rank completion, VAE, GNN — '
             'Eurostat IOT + air emissions accounts, NACE-64, 4 indicators, 3 seeds', fontsize=13)
for j, (pat, _, short) in enumerate(PATTERNS):
    plot_error(axes[0, j], pat, short)
    axes[0, j].set_title(short, fontsize=13)
    axes[0, j].set_ylabel('Impact MdAPE (log scale, fraction)' if j == 0 else '')
    if j == 0:
        axes[0, j].legend(loc='lower right', fontsize=8)
    plot_coverage(axes[1, j], pat, short)
    axes[1, j].set_title('')
    axes[1, j].set_ylabel('95% interval coverage (%)' if j == 0 else '')
    if j == 0:
        axes[1, j].legend(loc='center left', fontsize=8)
fig.tight_layout(rect=[0, 0, 1, 0.96])
fig.savefig(os.path.join(FIGURES, 'fig_eu_full_benchmark.png'), dpi=120)
plt.close(fig)

# ---- GSD sensitivity chart ----
fig, ax = plt.subplots(figsize=(8, 5.5))
combo_style = {
    ('MCAR', 0.20):  dict(color='tab:green',  label='MCAR @ 20%'),
    ('MCAR', 0.40):  dict(color='tab:red',    label='MCAR @ 40%'),
    ('FLOW', 0.10):  dict(color='tab:purple', label='FLOW @ 10% (closest-to-nominal condition)'),
    ('FLOW', 0.40):  dict(color='tab:brown',  label='FLOW @ 40%'),
    ('BLOCK', 0.20): dict(color='tab:blue',   label='BLOCK @ 20%'),
    ('BLOCK', 0.40): dict(color='tab:orange', label='BLOCK @ 40%'),
}
for (pat, rate), style in combo_style.items():
    d = gsd[(gsd.pattern == pat) & (np.isclose(gsd.rate, rate))].sort_values('gsd')
    ax.plot(d['gsd'], d['coverage'] * 100, 'o-', linewidth=2, markersize=7, **style)
ax.axhline(95, color='gray', linestyle=':', linewidth=1.5)
ax.text(1.2, 96, 'nominal 95%', color='gray', fontsize=9)
ax.set_ylim(0, 100)
ax.set_xlabel('Pedigree geometric standard deviation (GSD)')
ax.set_ylabel('95% interval coverage (%)')
ax.set_title('Calibration failure is not a GSD artifact')
ax.legend(loc='upper left', fontsize=9)
ax.grid(True, alpha=0.3)
fig.tight_layout()
fig.savefig(os.path.join(FIGURES, 'fig_gsd_sensitivity.png'), dpi=150)
plt.close(fig)

print('Figures written to', FIGURES)
for f in sorted(os.listdir(FIGURES)):
    print(' ', f)
