"""
Collect PEMS-BAY baseline test metrics from log files and print Markdown table.

Usage (from repo root):
    python collect_baseline_metrics.py

Reads:
    logs/graph-wavenet_pems-bay.log  (or stdout saved there)
    logs/dgcrn_pems-bay.log
    logs/dcrnn_pems-bay.log

GWN and DGCRN: script already prints a tabular summary at the end of all runs --
we extract `test|horizon` lines from there.

DCRNN: run 3x with --TEST_ONLY, so 3 blocks of per-horizon metrics appear in the log;
we parse all of them and compute mean+std here.
"""

import re
import numpy as np
import os

HORIZONS = {3: '15 min', 6: '30 min', 12: '60 min'}
EVAL_RE = re.compile(
    r'Evaluate best model on test data for horizon\s+(\d+),\s+'
    r'Test MAE:\s+([\d.]+),\s+Test MAPE:\s+([\d.]+),\s+Test RMSE:\s+([\d.]+)'
)
SUMMARY_RE = re.compile(
    r'^(\d+)\t([\d.]+)\t([\d.]+)\t([\d.]+)\t([\d.]+)\t([\d.]+)\t([\d.]+)'
)


def parse_summary_table(text, model_name):
    """Extract pre-computed mean/std rows from GWN or DGCRN log."""
    rows = {}
    in_table = False
    for line in text.splitlines():
        if 'test|horizon' in line:
            in_table = True
            continue
        if in_table:
            m = SUMMARY_RE.match(line.strip())
            if m:
                h = int(m.group(1))
                if h in HORIZONS:
                    rows[h] = {
                        'mae_mean': float(m.group(2)),
                        'rmse_mean': float(m.group(3)),
                        'mape_mean': float(m.group(4)),
                        'mae_std': float(m.group(5)),
                        'rmse_std': float(m.group(6)),
                        'mape_std': float(m.group(7)),
                    }
            elif line.strip() == '':
                in_table = False
    return rows


def parse_eval_blocks(text):
    """Parse 3 TEST_ONLY blocks from DCRNN log, return mean+std per horizon."""
    runs = {}
    for m in EVAL_RE.finditer(text):
        h, mae, mape, rmse = int(m.group(1)), float(m.group(2)), float(m.group(3)), float(m.group(4))
        runs.setdefault(h, {'mae': [], 'mape': [], 'rmse': []})
        runs[h]['mae'].append(mae)
        runs[h]['mape'].append(mape)
        runs[h]['rmse'].append(rmse)

    rows = {}
    for h, d in runs.items():
        if h in HORIZONS:
            rows[h] = {
                'mae_mean':  np.mean(d['mae']),
                'rmse_mean': np.mean(d['rmse']),
                'mape_mean': np.mean(d['mape']),
                'mae_std':   np.std(d['mae']),
                'rmse_std':  np.std(d['rmse']),
                'mape_std':  np.std(d['mape']),
            }
    return rows


def read_log(path):
    if not os.path.exists(path):
        print(f'  [MISSING] {path}')
        return None
    with open(path, encoding='utf-8', errors='replace') as f:
        return f.read()


def fmt(mean, std):
    return f'{mean:.4f} ± {std:.4f}'


def print_table(models):
    print('\n### PEMS-BAY Baseline Metrics (test set, mean ± std over 3 runs)\n')
    hdr = '| Model | Adaptive | MAE 15 min | MAE 30 min | MAE 60 min | RMSE 15 min | RMSE 60 min | MAPE 15 min | MAPE 60 min |'
    sep = '|---|---|---|---|---|---|---|---|---|'
    print(hdr)
    print(sep)
    for name, adapt, rows in models:
        if not rows:
            print(f'| {name} | {adapt} | (no data) |||||||')
            continue
        r3  = rows.get(3,  {})
        r6  = rows.get(6,  {})
        r12 = rows.get(12, {})
        def g(r, k): return fmt(r[k], r.get(k.replace('mean','std'), 0)) if r else '—'
        print(
            f'| {name} | {adapt} '
            f'| {g(r3,"mae_mean")} | {g(r6,"mae_mean")} | {g(r12,"mae_mean")} '
            f'| {g(r3,"rmse_mean")} | {g(r12,"rmse_mean")} '
            f'| {g(r3,"mape_mean")} | {g(r12,"mape_mean")} |'
        )
    print()


if __name__ == '__main__':
    models = []

    # DCRNN
    text = read_log('logs/dcrnn_pems-bay.log')
    rows = parse_eval_blocks(text) if text else {}
    models.append(('DCRNN', 'none', rows))

    # Graph WaveNet
    text = read_log('logs/graph-wavenet_pems-bay.log')
    rows = parse_summary_table(text, 'GWN') if text else {}
    models.append(('Graph WaveNet', 'static', rows))

    # DGCRN
    text = read_log('logs/dgcrn_pems-bay.log')
    rows = parse_summary_table(text, 'DGCRN') if text else {}
    models.append(('DGCRN', 'dynamic', rows))

    print_table(models)
