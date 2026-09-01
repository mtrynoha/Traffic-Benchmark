"""
Inference-time adjacency matrix sensitivity analysis for DGCRN.
Swaps predefined_A at inference without retraining.
The adaptive graph (emb1/emb2/lin1/lin2 hyper-GNN) stays frozen from the checkpoint.

Run from Traffic-Benchmark_Fork/:
    python eval_adj_sensitivity_dgcrn.py --checkpoint save/expDGCRN_metrla_0.pth
"""

import sys
import os
import argparse
import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'methods', 'DGCRN'))
from net import DGCRN
from util import load_dataset, load_adj, asym_adj, load_pickle, metric


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--checkpoint',   type=str, required=True)
    p.add_argument('--device',       type=str, default='cpu')
    p.add_argument('--adjdata',      type=str, default='data/sensor_graph/adj_mx.pkl')
    p.add_argument('--data',         type=str, default='data/METR-LA')
    p.add_argument('--num_nodes',    type=int, default=207)
    p.add_argument('--batch_size',   type=int, default=64)
    p.add_argument('--seed',         type=int, default=42)
    # must match training config
    p.add_argument('--gcn_depth',    type=int,   default=2)
    p.add_argument('--dropout',      type=float, default=0.3)
    p.add_argument('--subgraph_size',type=int,   default=20)
    p.add_argument('--node_dim',     type=int,   default=40)
    p.add_argument('--rnn_size',     type=int,   default=64)
    p.add_argument('--hyperGNN_dim', type=int,   default=16)
    p.add_argument('--layers',       type=int,   default=3)
    p.add_argument('--in_dim',       type=int,   default=2)
    p.add_argument('--seq_len',      type=int,   default=12)
    return p.parse_args()


def make_doubletransition(dense_adj):
    a = np.array(dense_adj, dtype=np.float32)
    return [asym_adj(a), asym_adj(a.T)]


def to_tensors(adj_list, device):
    return [torch.tensor(np.array(a), dtype=torch.float32).to(device) for a in adj_list]


def run_inference(model, dataloader, scaler, device, predefined_A):
    model.predefined_A = predefined_A
    model.eval()

    outputs = []
    realy = torch.Tensor(dataloader['y_test']).to(device)
    realy = realy.transpose(1, 3)[:, 0, :, :]  # (N, num_nodes, horizon)

    for x, y in dataloader['test_loader'].get_iterator():
        testx = torch.Tensor(x).to(device).transpose(1, 3)
        testy = torch.Tensor(y).to(device).transpose(1, 3)
        with torch.no_grad():
            preds = model(testx, ycl=testy)
            preds = preds.transpose(1, 3)
        outputs.append(preds.squeeze(dim=1))

    yhat = torch.cat(outputs, dim=0)[:realy.size(0)]

    per_horizon = {}
    for h in range(12):
        pred = scaler.inverse_transform(yhat[:, :, h])
        real = realy[:, :, h]
        mae, mape, rmse = metric(pred, real)
        per_horizon[h + 1] = dict(mae=mae, mape=mape * 100, rmse=rmse)
    return per_horizon


def main():
    args = parse_args()
    device = torch.device(args.device)
    rng = np.random.default_rng(args.seed)
    n = args.num_nodes

    adj_list = load_adj(args.adjdata)
    original_tensors = to_tensors(adj_list, device)

    _, _, raw_adj = load_pickle(args.adjdata)
    raw_adj = np.array(raw_adj, dtype=np.float32)

    density = (raw_adj > 0).mean()
    rand_adj = (rng.random((n, n)) < density).astype(np.float32)
    np.fill_diagonal(rand_adj, 1.0)
    perm = rng.permutation(n)
    perm_adj = raw_adj[np.ix_(perm, perm)]

    variants = {
        'original':              adj_list,
        'zeros':                 make_doubletransition(np.zeros((n, n))),
        'identity':              make_doubletransition(np.eye(n)),
        'random (same density)': make_doubletransition(rand_adj),
        'permuted nodes':        make_doubletransition(perm_adj),
        'fully connected':       make_doubletransition(np.ones((n, n))),
    }

    model = DGCRN(
        args.gcn_depth, n, device,
        predefined_A=original_tensors,
        dropout=args.dropout,
        subgraph_size=args.subgraph_size,
        node_dim=args.node_dim,
        middle_dim=2,
        seq_length=args.seq_len,
        in_dim=args.in_dim,
        out_dim=args.seq_len,
        layers=args.layers,
        list_weight=[0.05, 0.95, 0.95],
        tanhalpha=3,
        cl_decay_steps=2000,
        rnn_size=args.rnn_size,
        hyperGNN_dim=args.hyperGNN_dim,
    )
    model.to(device)
    model.load_state_dict(torch.load(args.checkpoint, map_location=device))
    print(f'Loaded: {args.checkpoint}')

    dataloader = load_dataset(args.data, args.batch_size, args.batch_size, args.batch_size)
    scaler = dataloader['scaler']

    all_results = {}
    for name, adj in variants.items():
        print(f'  evaluating: {name}...')
        all_results[name] = run_inference(model, dataloader, scaler, device, to_tensors(adj, device))

    baseline = all_results['original']
    col_w = 26
    print()
    print(f"{'Adj variant':<{col_w}}  {'MAE@15m':>8}  {'MAE@30m':>8}  {'MAE@60m':>8}  {'dMAE@15m':>10}  {'dMAE@30m':>10}  {'dMAE@60m':>10}")
    print('-' * (col_w + 2 + 8 * 3 + 6 + 10 * 3 + 6))
    for name, res in all_results.items():
        m3  = res[3]['mae']
        m6  = res[6]['mae']
        m12 = res[12]['mae']
        if name == 'original':
            delta_str = f"  {'---':>10}  {'---':>10}  {'---':>10}"
        else:
            delta_str = (f"  {m3  - baseline[3]['mae']:+10.4f}"
                         f"  {m6  - baseline[6]['mae']:+10.4f}"
                         f"  {m12 - baseline[12]['mae']:+10.4f}")
        print(f"{name:<{col_w}}  {m3:8.4f}  {m6:8.4f}  {m12:8.4f}{delta_str}")
    print()
    print('Note: adaptive graph (emb1/emb2 hyper-GNN) stays frozen -- deltas show predefined_A contribution.')

    from perturb import evaluate_operational
    eval_fn = lambda dense: run_inference(
        model, dataloader, scaler, device,
        to_tensors(make_doubletransition(dense), device))
    evaluate_operational(eval_fn, raw_adj, baseline, model_name='DGCRN',
                         modes=('edge_removal', 'weight_noise', 'targeted_betweenness'),
                         rates=(0.1, 0.2, 0.3),
                         csv_path='results_dgcrn_operational.csv')


if __name__ == '__main__':
    main()