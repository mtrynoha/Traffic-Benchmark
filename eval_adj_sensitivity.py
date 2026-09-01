"""
Inference-time adjacency matrix sensitivity analysis for Graph-WaveNet.

Substitutes the fixed adjacency matrix with alternatives at inference time,
without any retraining. The learned adaptive adj (nodevec1/nodevec2) stays
frozen from the checkpoint, so deltas isolate the fixed-graph contribution.

Run from Traffic-Benchmark_Fork/:
    python eval_adj_sensitivity.py --checkpoint garage/metr_exp1_best_2.74.pth
"""

import sys
import os
import argparse
import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'methods', 'Graph-WaveNet'))
import util
from util import asym_adj
from model import gwnet

REPORT_HORIZONS = {3: '15min', 6: '30min', 12: '60min'}


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--checkpoint', type=str, required=True,
                   help='Path to .pth checkpoint, e.g. garage/metr_exp1_best_2.74.pth')
    p.add_argument('--device',    type=str, default='cuda:0')
    p.add_argument('--adjdata',   type=str, default='data/sensor_graph/adj_mx.pkl')
    p.add_argument('--data',      type=str, default='data/METR-LA')
    p.add_argument('--num_nodes', type=int, default=207)
    p.add_argument('--nhid',      type=int, default=32)
    p.add_argument('--in_dim',    type=int, default=2)
    p.add_argument('--dropout',   type=float, default=0.3)
    p.add_argument('--batch_size',type=int, default=64)
    p.add_argument('--seed',      type=int, default=42)
    return p.parse_args()


def make_doubletransition(dense_adj):
    """Return [D^{-1}A, D^{-1}A^T] (same format as util.load_adj doubletransition)."""
    a = np.array(dense_adj, dtype=np.float32)
    return [asym_adj(a), asym_adj(a.T)]


def to_tensors(adj_list, device):
    return [torch.tensor(np.array(a), dtype=torch.float32).to(device) for a in adj_list]


def run_inference(model, dataloader, scaler, device, supports_tensors):
    model.supports = supports_tensors
    model.eval()

    outputs = []
    realy = torch.Tensor(dataloader['y_test']).to(device)
    realy = realy.transpose(1, 3)[:, 0, :, :]

    for x, _ in dataloader['test_loader'].get_iterator():
        testx = torch.Tensor(x).to(device).transpose(1, 3)
        with torch.no_grad():
            preds = model(testx).transpose(1, 3)
        outputs.append(preds.squeeze())

    yhat = torch.cat(outputs, dim=0)[:realy.size(0)]

    per_horizon = {}
    for h in range(12):
        pred = scaler.inverse_transform(yhat[:, :, h])
        real = realy[:, :, h]
        mae, mape, rmse = util.metric(pred, real)
        per_horizon[h + 1] = dict(mae=mae, mape=mape * 100, rmse=rmse)
    return per_horizon


def main():
    args = parse_args()
    device = torch.device(args.device)
    rng = np.random.default_rng(args.seed)
    n = args.num_nodes

    # ── Load original adjacency ───────────────────────────────────────────────
    _, _, adj_list = util.load_adj(args.adjdata, 'doubletransition')
    original_tensors = to_tensors(adj_list, device)

    # Raw adj matrix for building variants (reload from pickle directly)
    _, _, raw_adj = util.load_pickle(args.adjdata)
    raw_adj = np.array(raw_adj, dtype=np.float32)

    # ── Build adjacency variants ──────────────────────────────────────────────
    density = (raw_adj > 0).mean()

    rand_adj = (rng.random((n, n)) < density).astype(np.float32)
    np.fill_diagonal(rand_adj, 1.0)

    perm = rng.permutation(n)
    perm_adj = raw_adj[np.ix_(perm, perm)]

    variants = {
        'original':             adj_list,
        'zeros':                make_doubletransition(np.zeros((n, n))),
        'identity':             make_doubletransition(np.eye(n)),
        'random (same density)': make_doubletransition(rand_adj),
        'permuted nodes':       make_doubletransition(perm_adj),
        'fully connected':      make_doubletransition(np.ones((n, n))),
    }

    # ── Load model once ───────────────────────────────────────────────────────
    model = gwnet(
        device, n, args.dropout,
        supports=original_tensors, gcn_bool=True, addaptadj=True,
        aptinit=original_tensors[0],
        in_dim=args.in_dim,
        residual_channels=args.nhid, dilation_channels=args.nhid,
        skip_channels=args.nhid * 8, end_channels=args.nhid * 16,
    )
    model.to(device)
    model.load_state_dict(torch.load(args.checkpoint, map_location=device))
    print(f'Loaded: {args.checkpoint}')

    dataloader = util.load_dataset(args.data, args.batch_size, args.batch_size, args.batch_size)
    scaler = dataloader['scaler']

    # ── Evaluate each variant ─────────────────────────────────────────────────
    all_results = {}
    for name, adj in variants.items():
        print(f'  evaluating: {name}...')
        all_results[name] = run_inference(model, dataloader, scaler, device, to_tensors(adj, device))

    # ── Print result table ────────────────────────────────────────────────────
    baseline = all_results['original']
    col_w = 26

    print()
    print(f"{'Adj variant':<{col_w}}  "
          f"{'MAE@15m':>8}  {'MAE@30m':>8}  {'MAE@60m':>8}  "
          f"{'dMAE@15m':>10}  {'dMAE@30m':>10}  {'dMAE@60m':>10}")
    print('-' * (col_w + 2 + 8*3 + 6 + 10*3 + 6))

    for name, res in all_results.items():
        m3  = res[3]['mae']
        m6  = res[6]['mae']
        m12 = res[12]['mae']
        if name == 'original':
            delta_str = f"  {'---':>10}  {'---':>10}  {'---':>10}"
        else:
            d3  = m3  - baseline[3]['mae']
            d6  = m6  - baseline[6]['mae']
            d12 = m12 - baseline[12]['mae']
            delta_str = f"  {d3:+10.4f}  {d6:+10.4f}  {d12:+10.4f}"
        print(f"{name:<{col_w}}  {m3:8.4f}  {m6:8.4f}  {m12:8.4f}{delta_str}")

    print()
    print('Note: adaptive adj (nodevec1/nodevec2) stays frozen -- deltas show fixed-graph contribution.')

    from perturb import evaluate_operational
    eval_fn = lambda dense: run_inference(
        model, dataloader, scaler, device,
        to_tensors(make_doubletransition(dense), device))
    evaluate_operational(eval_fn, raw_adj, baseline, model_name='GWN',
                         modes=('edge_removal', 'weight_noise', 'targeted_betweenness'),
                         rates=(0.1, 0.2, 0.3),
                         csv_path='results_gwn_operational.csv')


if __name__ == '__main__':
    main()
