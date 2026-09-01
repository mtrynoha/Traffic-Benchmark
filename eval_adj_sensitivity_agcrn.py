"""
Inference-time graph sensitivity analysis for AGCRN (PEMS-BAY).

AGCRN has NO predefined graph: it LEARNS a dense adjacency
    S = softmax(relu(E @ E^T))
inside every AVWGCN from the node embeddings E. We freeze the trained weights
(incl. the node-embedding-derived conv weights/bias) and substitute ONLY the
graph S at inference -- the exact analog of "freeze adaptive part, perturb the
graph" used for DCRNN/GWN/DGCRN, applied to AGCRN's *learned* graph.

Two framings (both requested):
  (a) perturb the learned graph -> real robustness curve (this script);
  (b) AGCRN as the fully-adaptive endpoint: it has no predefined graph, so it is
      immune by construction to the predefined-graph corruption the other 3 suffer
      (reported as a conceptual point in the paper).

Run from Traffic-Benchmark_Fork/:
    python eval_adj_sensitivity_agcrn.py --checkpoint save/agcrn_pemsbay_run0.pth
"""

import sys, os, argparse
import numpy as np
import torch
import torch.nn.functional as F

AGCRN_DIR = os.path.join(os.path.dirname(__file__), 'methods', 'AGCRN')
sys.path.insert(0, AGCRN_DIR)

from model.AGCRN import AGCRN
import model.AGCN as AGCN
from lib.dataloader import get_dataloader
from lib.metrics import All_Metrics

REPORT = {3: '15min', 6: '30min', 12: '60min'}


# ── Class-level hook: make every AVWGCN use an externally supplied graph ───────
_OVERRIDE = {'S': None}      # if a tensor, used as the learned graph; else recompute

def patched_forward(self, x, node_embeddings):
    node_num = node_embeddings.shape[0]
    if _OVERRIDE['S'] is None:
        supports = F.softmax(F.relu(torch.mm(node_embeddings, node_embeddings.transpose(0, 1))), dim=1)
    else:
        supports = _OVERRIDE['S']
    support_set = [torch.eye(node_num).to(supports.device), supports]
    for k in range(2, self.cheb_k):
        support_set.append(torch.matmul(2 * supports, support_set[-1]) - support_set[-2])
    supports = torch.stack(support_set, dim=0)
    weights = torch.einsum('nd,dkio->nkio', node_embeddings, self.weights_pool)
    bias = torch.matmul(node_embeddings, self.bias_pool)
    x_g = torch.einsum("knm,bmc->bknc", supports, x)
    x_g = x_g.permute(0, 2, 1, 3)
    x_gconv = torch.einsum('bnki,nkio->bno', x_g, weights) + bias
    return x_gconv

AGCN.AVWGCN.forward = patched_forward


def build_args(ckpt, device, num_nodes, dataset_dir):
    a = argparse.Namespace()
    a.num_nodes = num_nodes; a.input_dim = 2; a.output_dim = 1
    a.embed_dim = 10; a.rnn_units = 64; a.num_layers = 2; a.cheb_k = 2
    a.horizon = 12; a.lag = 12; a.default_graph = True
    a.batch_size = 64; a.device = device
    a.dataset_dir = dataset_dir; a.dataset = os.path.basename(dataset_dir.rstrip('/'))
    a.normalizer = 'std'; a.tod = False; a.column_wise = False
    a.mae_thresh = 0.0; a.mape_thresh = 0.0
    return a


def run_inference(model, test_loader, scaler, device):
    model.eval()
    y_pred, y_true = [], []
    with torch.no_grad():
        for data, target in test_loader:
            data = data[..., :model.input_dim]
            label = target[..., :model.output_dim]
            out = model(data, target, teacher_forcing_ratio=0)
            y_pred.append(out); y_true.append(label)
    y_true = torch.cat(y_true, dim=0)
    y_pred = scaler.inverse_transform(torch.cat(y_pred, dim=0))   # (N, horizon, nodes, 1)
    per_h = {}
    for t in range(y_true.shape[1]):
        mae, rmse, mape, _, _ = All_Metrics(y_pred[:, t, ...], y_true[:, t, ...], 0.0, 0.0)
        per_h[t + 1] = dict(mae=float(mae), rmse=float(rmse), mape=float(mape) * 100)
    return per_h


def renorm_rows(S):
    s = S.sum(dim=1, keepdim=True)
    s[s == 0] = 1.0
    return S / s


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--checkpoint', required=True)
    p.add_argument('--device', default='cuda:0')
    p.add_argument('--data', default='data/PEMS-BAY', help='dataset_dir')
    p.add_argument('--num_nodes', type=int, default=325)
    p.add_argument('--seed', type=int, default=42)
    args = p.parse_args()
    suffix = 'pemsbay' if 'PEMS' in args.data.upper() else ('metrla' if 'METR' in args.data.upper() else os.path.basename(args.data.rstrip('/')).lower())

    if torch.cuda.is_available():
        torch.cuda.set_device(int(args.device[5]))
        device = torch.device(args.device)
    else:
        device = torch.device('cpu')
    rng = np.random.default_rng(args.seed)
    n = args.num_nodes

    a = build_args(args.checkpoint, args.device, n, args.data)
    model = AGCRN(a).to(device)
    model.load_state_dict(torch.load(args.checkpoint, map_location=device))
    print(f'Loaded: {args.checkpoint}')

    _, _, test_loader, scaler = get_dataloader(a, normalizer='std', tod=False, dow=False, weather=False, single=False)

    # clean learned graph from trained embeddings
    E = model.node_embeddings.detach()
    S_clean = F.softmax(F.relu(torch.mm(E, E.transpose(0, 1))), dim=1).to(device)

    eye = torch.eye(n, device=device)
    rand_S = renorm_rows(F.relu(torch.tensor(rng.standard_normal((n, n)), dtype=torch.float32, device=device)))
    perm = rng.permutation(n)
    perm_S = S_clean[np.ix_(perm, perm)]
    uniform_S = torch.full((n, n), 1.0 / n, device=device)

    variants = {
        'original (learned)':    None,            # recompute = clean
        'zeros':                 torch.zeros((n, n), device=device),
        'identity':              eye.clone(),
        'random (renorm)':       rand_S,
        'permuted nodes':        perm_S,
        'uniform (fully conn.)': uniform_S,
    }

    results = {}
    for name, S in variants.items():
        _OVERRIDE['S'] = S
        print(f'  evaluating: {name}...')
        results[name] = run_inference(model, test_loader, scaler, device)
    _OVERRIDE['S'] = None

    base = results['original (learned)']
    cw = 24
    print()
    print(f"{'Learned-graph variant':<{cw}}  {'MAE@15m':>8}  {'MAE@30m':>8}  {'MAE@60m':>8}  {'dMAE@15m':>10}  {'dMAE@30m':>10}  {'dMAE@60m':>10}")
    print('-' * (cw + 2 + 8 * 3 + 6 + 10 * 3 + 6))
    for name, r in results.items():
        m3, m6, m12 = r[3]['mae'], r[6]['mae'], r[12]['mae']
        if name.startswith('original'):
            d = f"  {'---':>10}  {'---':>10}  {'---':>10}"
        else:
            d = f"  {m3-base[3]['mae']:+10.4f}  {m6-base[6]['mae']:+10.4f}  {m12-base[12]['mae']:+10.4f}"
        print(f"{name:<{cw}}  {m3:8.4f}  {m6:8.4f}  {m12:8.4f}{d}")
    print()
    print('Note: AGCRN has NO predefined graph; perturbations above corrupt its LEARNED graph S=softmax(relu(EE^T)).')

    # ── Operational on the learned graph: edge_removal + weight_noise (3 seeds) ─
    print()
    print('[AGCRN] operational perturbations on learned graph (mean over 3 seeds)')
    print('mode             rate | dMAE@3  dMAE@6  dMAE@12')
    print('-' * 52)
    rows = []
    Scpu = S_clean.cpu().numpy()
    for rate in (0.1, 0.2, 0.3):
        for mode in ('edge_removal', 'weight_noise'):
            d3 = d6 = d12 = 0.0
            for rep in range(3):
                r2 = np.random.default_rng(args.seed + rep)
                Sm = Scpu.copy()
                if mode == 'edge_removal':
                    mask = r2.random(Sm.shape) < rate
                    Sm[mask] = 0.0
                else:  # weight_noise: multiplicative log-normal
                    Sm = Sm * np.exp(r2.normal(0.0, rate, Sm.shape).astype(np.float32))
                St = renorm_rows(torch.tensor(Sm, dtype=torch.float32, device=device))
                _OVERRIDE['S'] = St
                rr = run_inference(model, test_loader, scaler, device)
                d3 += rr[3]['mae'] - base[3]['mae']
                d6 += rr[6]['mae'] - base[6]['mae']
                d12 += rr[12]['mae'] - base[12]['mae']
            d3, d6, d12 = d3 / 3, d6 / 3, d12 / 3
            print(f'{mode:<16} {int(rate*100):>3}% | {d3:+.3f}  {d6:+.3f}  {d12:+.3f}')
            rows.append((mode, rate, d3, d6, d12))
    _OVERRIDE['S'] = None

    import csv
    csv_name = f'results_agcrn_operational_{suffix}.csv'
    with open(csv_name, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['mode', 'rate', 'dMAE@3', 'dMAE@6', 'dMAE@12'])
        for row in rows:
            w.writerow(row)
    print(f'\nsaved: {csv_name}')
    print('Note: targeted-betweenness N/A -- learned graph is dense (softmax), no edge bottlenecks to attack.')


if __name__ == '__main__':
    main()
