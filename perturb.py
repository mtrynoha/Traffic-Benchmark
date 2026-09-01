"""
Operational + targeted adjacency perturbations for inference-time robustness analysis.

Model-agnostic: produces perturbed DENSE adjacency matrices from the original
raw_adj. Feed each one through your existing model-specific support builder
(make_doubletransition for GWN/DGCRN, make_dcrnn_supports for DCRNN) and your
existing run_inference(). The learned adaptive graph stays frozen, exactly as in
the structural-substitution experiment, so deltas are directly comparable.

INVARIANT: every perturbation preserves the diagonal (self-loops). This keeps the
physics honest (a node always sees its own history) and guarantees no all-zero row,
so random-walk / asym normalization never divides by zero.

Experiments
-----------
  Exp 2 (operational, stochastic):  edge_removal | weight_noise | node_isolation
  Exp 3 (targeted, deterministic):  targeted_betweenness | targeted_weight
                                     (contrast against random edge_removal)

Usage: see integration snippets at the bottom of this file.
"""

import csv
import numpy as np


# ── Helpers ───────────────────────────────────────────────────────────────────

def _off_diag_mask(adj):
    """Boolean mask of existing off-diagonal (directed) edges."""
    n = adj.shape[0]
    return (adj > 0) & ~np.eye(n, dtype=bool)


# ── Exp 2: operational perturbations (stochastic) ───────────────────────────────

def edge_removal(adj, rate, rng):
    """Remove `rate` fraction of off-diagonal edges at random (link failures)."""
    adj = adj.astype(np.float32).copy()
    idx = np.argwhere(_off_diag_mask(adj))
    k = int(round(len(idx) * rate))
    if k > 0:
        chosen = rng.choice(len(idx), size=k, replace=False)
        for i, j in idx[chosen]:
            adj[i, j] = 0.0
    return adj


def weight_noise(adj, sigma, rng):
    """Multiply off-diagonal weights by lognormal(0, sigma) noise (congestion/weather).
    Positivity preserved; diagonal untouched. `sigma` plays the role of `rate`."""
    adj = adj.astype(np.float32).copy()
    mask = _off_diag_mask(adj)
    noise = rng.lognormal(mean=0.0, sigma=float(sigma), size=adj.shape).astype(np.float32)
    adj[mask] = adj[mask] * noise[mask]
    return adj


def node_isolation(adj, rate, rng):
    """Isolate `rate` fraction of nodes: zero their off-diagonal row & col
    (a junction taken out of service). Self-loops restored afterwards."""
    adj = adj.astype(np.float32).copy()
    n = adj.shape[0]
    diag = np.diag(adj).copy()
    k = int(round(n * rate))
    if k > 0:
        failed = rng.choice(n, size=k, replace=False)
        adj[failed, :] = 0.0
        adj[:, failed] = 0.0
    np.fill_diagonal(adj, diag)  # always keep self-loops
    return adj


# ── Exp 3: targeted attack (deterministic) ──────────────────────────────────────

def targeted_edge_removal(adj, rate, rng=None, strategy='betweenness'):
    """Remove the `rate` fraction of most-critical off-diagonal edges first.

    strategy='betweenness' : edge-betweenness on a DiGraph with cost = 1/weight
                             (strong similarity -> short path cost). Needs networkx.
    strategy='weight'      : remove the strongest edges first (no extra deps).
    """
    adj = adj.astype(np.float32).copy()
    idx = np.argwhere(_off_diag_mask(adj))
    if len(idx) == 0:
        return adj

    if strategy == 'weight':
        scores = adj[tuple(idx.T)]                       # importance proxy = weight
    elif strategy == 'betweenness':
        import networkx as nx
        G = nx.DiGraph()
        for i, j in idx:
            G.add_edge(int(i), int(j), weight=1.0 / float(adj[i, j]))
        eb = nx.edge_betweenness_centrality(G, weight='weight')
        scores = np.array([eb.get((int(i), int(j)), 0.0) for i, j in idx])
    else:
        raise ValueError(f'unknown strategy: {strategy}')

    order = np.argsort(-scores)                          # descending importance
    k = int(round(len(idx) * rate))
    for t in range(k):
        i, j = idx[order[t]]
        adj[i, j] = 0.0
    return adj


# ── Driver ──────────────────────────────────────────────────────────────────────

_STOCHASTIC = {'edge_removal', 'weight_noise', 'node_isolation'}

_GENERATORS = {
    'edge_removal':         edge_removal,
    'weight_noise':         weight_noise,
    'node_isolation':       node_isolation,
    'targeted_weight':      lambda a, r, rng: targeted_edge_removal(a, r, strategy='weight'),
    'targeted_betweenness': lambda a, r, rng: targeted_edge_removal(a, r, strategy='betweenness'),
}


def evaluate_operational(eval_fn, raw_adj, baseline, *,
                         model_name='model',
                         modes=('edge_removal', 'weight_noise',
                                'node_isolation', 'targeted_betweenness'),
                         rates=(0.1, 0.2, 0.3),
                         n_repeats=3, base_seed=42,
                         horizons=(3, 6, 12),
                         csv_path=None, verbose=True):
    """
    eval_fn   : callable(dense_adj) -> {h: {'mae','mape','rmse'}}  (wraps make_*+run_inference)
    raw_adj   : original dense adjacency (np.ndarray)
    baseline  : per-horizon dict from eval_fn(raw_adj)  -- compute via the SAME path!
    Returns   : list of result rows (also written to csv_path if given).

    Stochastic modes are averaged over n_repeats independent seeds (mean +/- std).
    Deterministic targeted modes run once.
    """
    rows = []
    if verbose:
        hdr = f"{'mode':<22}{'rate':>6} | " + "  ".join(f"dMAE@{h}" for h in horizons)
        print(f"\n[{model_name}] operational + targeted perturbations")
        print(hdr)
        print('-' * len(hdr))

    for mode in modes:
        gen = _GENERATORS[mode]
        reps = n_repeats if mode in _STOCHASTIC else 1
        for rate in rates:
            maes = {h: [] for h in horizons}
            for r in range(reps):
                rng = np.random.default_rng(base_seed + r)
                res = eval_fn(gen(raw_adj, rate, rng))
                for h in horizons:
                    maes[h].append(res[h]['mae'])
            row = {'model': model_name, 'mode': mode, 'rate': rate, 'n_repeats': reps}
            for h in horizons:
                arr = np.array(maes[h], dtype=float)
                row[f'mae@{h}']     = float(arr.mean())
                row[f'mae_std@{h}'] = float(arr.std())
                row[f'dmae@{h}']    = float(arr.mean() - baseline[h]['mae'])
            rows.append(row)
            if verbose:
                deltas = "  ".join(f"{row[f'dmae@{h}']:+7.3f}" for h in horizons)
                print(f"{mode:<22}{rate:>6.0%} | {deltas}")

    if csv_path:
        fieldnames = (['model', 'mode', 'rate', 'n_repeats']
                      + [f'mae@{h}' for h in horizons]
                      + [f'mae_std@{h}' for h in horizons]
                      + [f'dmae@{h}' for h in horizons])
        with open(csv_path, 'w', newline='') as f:
            w = csv.DictWriter(f, fieldnames=fieldnames)
            w.writeheader()
            w.writerows(rows)
        if verbose:
            print(f"\nsaved: {csv_path}")
    return rows


# ──────────────────────────────────────────────────────────────────────────────
# INTEGRATION SNIPPETS  (add ~6 lines to each existing eval_adj_sensitivity*.py)
# ──────────────────────────────────────────────────────────────────────────────
#
# --- Graph-WaveNet (eval_adj_sensitivity.py) -----------------------------------
#   from perturb import evaluate_operational
#   # baseline THROUGH THE SAME PATH as perturbed (clean deltas):
#   baseline = run_inference(model, dataloader, scaler, device,
#                            to_tensors(make_doubletransition(raw_adj), device))
#   eval_fn = lambda dense: run_inference(
#       model, dataloader, scaler, device,
#       to_tensors(make_doubletransition(dense), device))
#   evaluate_operational(eval_fn, raw_adj, baseline, model_name='GWN',
#                        csv_path='results_gwn_operational.csv')
#
# --- DGCRN (eval_adj_sensitivity_dgcrn.py) -------------------------------------
#   from perturb import evaluate_operational
#   baseline = run_inference(model, dataloader, scaler, device,
#                            to_tensors(make_doubletransition(raw_adj), device))
#   eval_fn = lambda dense: run_inference(
#       model, dataloader, scaler, device,
#       to_tensors(make_doubletransition(dense), device))
#   evaluate_operational(eval_fn, raw_adj, baseline, model_name='DGCRN',
#                        csv_path='results_dgcrn_operational.csv')
#
# --- DCRNN (eval_adj_sensitivity_dcrnn.py) -------------------------------------
#   from perturb import evaluate_operational
#   baseline = run_inference(model, data, scaler, make_dcrnn_supports(raw_adj), cfg)
#   eval_fn = lambda dense: run_inference(
#       model, data, scaler, make_dcrnn_supports(dense), cfg)
#   evaluate_operational(eval_fn, raw_adj, baseline, model_name='DCRNN',
#                        csv_path='results_dcrnn_operational.csv')
#   # NOTE: DCRNN has no adaptive graph -> deltas show full degradation.
# ──────────────────────────────────────────────────────────────────────────────