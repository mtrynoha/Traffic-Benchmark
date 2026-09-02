"""
Plot adjacency sensitivity results for GWN, DGCRN, DCRNN.
Produces three figures:
  1. Bar chart  — dMAE@60min by adj variant, models side-by-side
  2. Line chart — MAE vs horizon for each variant (one subplot per model)
  3. Heatmap    — models x variants, colour = dMAE@60min
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')          # headless rendering
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import os

OUT_DIR = 'figures'
os.makedirs(OUT_DIR, exist_ok=True)

# ── Raw results ───────────────────────────────────────────────────────────────
# Each entry: (MAE@15m, MAE@30m, MAE@60m)

RESULTS = {
    'GWN': {
        'original':              (2.6955, 3.0843, 3.5266),
        'zeros':                 (6.7632, 7.8732, 8.6782),
        'identity':              (3.5403, 4.3123, 5.3197),
        'random':                (3.8300, 4.9988, 6.0993),
        'permuted':              (3.7744, 4.7948, 5.7114),
        'fully connected':       (4.0515, 5.6173, 6.7521),
    },
    'DGCRN': {
        'original':              (2.6211, 2.9915, 3.4664),
        'zeros':                 (8.2063, 8.6913, 9.2213),
        'identity':              (3.3167, 3.9802, 4.9336),
        'random':                (3.6324, 4.5003, 5.4297),
        'permuted':              (3.4803, 4.3311, 5.3017),
        'fully connected':       (3.7321, 4.7678, 5.8855),
    },
    'DCRNN': {
        'original':              (2.7544, 3.1571, 3.6343),
        'zeros':                 (6.9785, 7.4835, 7.8175),
        'identity':              (3.6097, 4.5129, 5.6551),
        'random':                (4.0496, 5.2159, 6.5698),
        'permuted':              (4.4684, 5.6965, 6.9374),
        'fully connected':       (4.2025, 5.2856, 6.3922),
    },
}

MODELS    = ['GWN', 'DGCRN', 'DCRNN']
VARIANTS  = ['zeros', 'identity', 'random', 'permuted', 'fully connected']
HORIZONS  = [15, 30, 60]
H_IDX     = [0, 1, 2]    # index into the MAE tuple

MODEL_COLORS  = {'GWN': '#4C72B0', 'DGCRN': '#DD8452', 'DCRNN': '#55A868'}
VARIANT_STYLES = {
    'zeros':           ('-',  'o'),
    'identity':        ('--', 's'),
    'random':          ('-.',  '^'),
    'permuted':        (':',  'D'),
    'fully connected': ('-',  'P'),
}
VARIANT_COLORS = ['#e41a1c', '#377eb8', '#4daf4a', '#984ea3', '#ff7f00']


def delta(model, variant, h_idx):
    base = RESULTS[model]['original'][h_idx]
    return RESULTS[model][variant][h_idx] - base


# ── Figure 1: grouped bar chart — dMAE@60min ─────────────────────────────────
fig, ax = plt.subplots(figsize=(9, 5))

n_var = len(VARIANTS)
n_mod = len(MODELS)
x     = np.arange(n_var)
width = 0.25
offsets = [-width, 0, width]

for i, (model, off) in enumerate(zip(MODELS, offsets)):
    vals = [delta(model, v, 2) for v in VARIANTS]
    bars = ax.bar(x + off, vals, width, label=model,
                  color=MODEL_COLORS[model], edgecolor='white', linewidth=0.5)
    for bar, val in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.04,
                f'{val:.2f}', ha='center', va='bottom', fontsize=7.5)

ax.set_xticks(x)
ax.set_xticklabels(VARIANTS, rotation=18, ha='right', fontsize=9)
ax.set_ylabel('dMAE@60min  (vs original adj)', fontsize=10)
ax.set_title('Adjacency Substitution Degradation at 60-min Horizon\n(METR-LA inference-time sensitivity)', fontsize=11)
ax.legend(fontsize=9)
ax.yaxis.set_minor_locator(mticker.AutoMinorLocator())
ax.grid(axis='y', alpha=0.3, linestyle='--')
ax.set_ylim(0, max(delta(m, v, 2) for m in MODELS for v in VARIANTS) * 1.18)
fig.tight_layout()
fig.savefig(f'{OUT_DIR}/fig1_bar_dMAE60.png', dpi=150)
plt.close(fig)
print('Saved fig1_bar_dMAE60.png')


# ── Figure 2: line charts — MAE vs horizon per model ─────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(14, 4.5), sharey=False)

for ax, model in zip(axes, MODELS):
    # original baseline as shaded band reference
    base_vals = [RESULTS[model]['original'][hi] for hi in H_IDX]
    ax.plot(HORIZONS, base_vals, 'k--', linewidth=1.8, label='original', zorder=5)

    for vi, variant in enumerate(VARIANTS):
        vals = [RESULTS[model][variant][hi] for hi in H_IDX]
        ls, mk = VARIANT_STYLES[variant]
        ax.plot(HORIZONS, vals,
                linestyle=ls, marker=mk, markersize=6,
                color=VARIANT_COLORS[vi], label=variant, linewidth=1.4)

    ax.set_title(model, fontsize=12, fontweight='bold')
    ax.set_xlabel('Prediction horizon (min)', fontsize=9)
    ax.set_ylabel('MAE', fontsize=9)
    ax.set_xticks(HORIZONS)
    ax.grid(alpha=0.25, linestyle='--')
    ax.legend(fontsize=7.5, loc='upper left')

fig.suptitle('MAE vs Horizon under Adjacency Substitution  (METR-LA)', fontsize=12, y=1.01)
fig.tight_layout()
fig.savefig(f'{OUT_DIR}/fig2_line_MAE_horizon.png', dpi=150, bbox_inches='tight')
plt.close(fig)
print('Saved fig2_line_MAE_horizon.png')


# ── Figure 3: heatmap — models x variants, dMAE@60min ────────────────────────
data_hm = np.array([[delta(m, v, 2) for v in VARIANTS] for m in MODELS])

fig, ax = plt.subplots(figsize=(8, 3.2))
im = ax.imshow(data_hm, aspect='auto', cmap='YlOrRd', vmin=0)

ax.set_xticks(range(n_var))
ax.set_xticklabels(VARIANTS, fontsize=9, rotation=20, ha='right')
ax.set_yticks(range(n_mod))
ax.set_yticklabels(MODELS, fontsize=10)

for i in range(n_mod):
    for j in range(n_var):
        val = data_hm[i, j]
        colour = 'white' if val > data_hm.max() * 0.65 else 'black'
        ax.text(j, i, f'+{val:.2f}', ha='center', va='center',
                fontsize=9, color=colour, fontweight='bold')

cbar = fig.colorbar(im, ax=ax, fraction=0.03, pad=0.02)
cbar.set_label('dMAE@60min', fontsize=9)
ax.set_title('Adjacency Substitution Heatmap  —  dMAE at 60-min Horizon\n(METR-LA, inference-time, no retraining)', fontsize=11)
fig.tight_layout()
fig.savefig(f'{OUT_DIR}/fig3_heatmap_dMAE60.png', dpi=150)
plt.close(fig)
print('Saved fig3_heatmap_dMAE60.png')

print(f'\nAll figures written to ./{OUT_DIR}/')
