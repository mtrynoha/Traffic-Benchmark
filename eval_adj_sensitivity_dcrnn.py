"""
Inference-time adjacency matrix sensitivity analysis for DCRNN.
Replaces DCGRUCell._supports (sparse diffusion matrices) across all encoder/decoder
cells without retraining. The GRU weight matrices stay frozen from the checkpoint.

IMPORTANT: DCRNN uses lazy weight initialization (LayerParams). The model must
complete one forward pass before load_state_dict, otherwise no weights exist yet.

Run from Traffic-Benchmark_Fork/:
    python eval_adj_sensitivity_dcrnn.py --checkpoint models/la_best.tar
"""

import sys
import os
import argparse
import numpy as np
import torch
import yaml

DCRNN_DIR = os.path.join(os.path.dirname(__file__), 'methods', 'DCRNN')
sys.path.insert(0, DCRNN_DIR)

from lib import utils as dcrnn_utils
from model.pytorch.dcrnn_model import DCRNNModel
from model.pytorch.dcrnn_cell import DCGRUCell
from model.pytorch.utils import metric


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--checkpoint', type=str, default='models/la_best.tar')
    p.add_argument('--config',     type=str, default='methods/DCRNN/data/model/dcrnn_la.yaml')
    p.add_argument('--adjdata',    type=str, default='data/sensor_graph/adj_mx.pkl')
    p.add_argument('--device',     type=str, default='cuda:0')
    p.add_argument('--seed',       type=int, default=42)
    return p.parse_args()


# ── Adjacency helpers ─────────────────────────────────────────────────────────

def to_float32_sparse(t):
    """scipy diags upcasts to float64 -- recast values to match model weights."""
    t = t.coalesce()
    return torch.sparse_coo_tensor(t.indices(), t.values().float(), t.shape)


def make_dcrnn_supports(adj_dense):
    """Build [fwd_sparse, bwd_sparse] dual random-walk supports for a dense adj."""
    a = np.array(adj_dense, dtype=np.float32)
    rw_fwd = dcrnn_utils.calculate_random_walk_matrix(a).T
    rw_bwd = dcrnn_utils.calculate_random_walk_matrix(a.T).T
    return [
        to_float32_sparse(DCGRUCell._build_sparse_matrix(rw_fwd.tocoo())),
        to_float32_sparse(DCGRUCell._build_sparse_matrix(rw_bwd.tocoo())),
    ]


def set_supports(model, supports):
    """Replace _supports in every DCGRUCell in encoder and decoder."""
    for cell in model.encoder_model.dcgru_layers:
        cell._supports = supports
    for cell in model.decoder_model.dcgru_layers:
        cell._supports = supports


# ── Data helpers ──────────────────────────────────────────────────────────────

def prepare_x(x_np, seq_len, num_nodes, input_dim):
    x = torch.from_numpy(x_np).float()          # (batch, seq, nodes, dim)
    x = x.permute(1, 0, 2, 3)                   # (seq, batch, nodes, dim)
    batch = x.size(1)
    return x.reshape(seq_len, batch, num_nodes * input_dim)


def prepare_y(y_np, horizon, num_nodes, output_dim=1):
    y = torch.from_numpy(y_np).float()           # (batch, horizon, nodes, dim)
    y = y.permute(1, 0, 2, 3)                   # (horizon, batch, nodes, dim)
    batch = y.size(1)
    return y[..., :output_dim].reshape(horizon, batch, num_nodes * output_dim)


# ── Inference ─────────────────────────────────────────────────────────────────

def run_inference(model, data, scaler, supports, cfg, device):
    set_supports(model, supports)
    model.eval()

    seq_len    = cfg['model']['seq_len']
    horizon    = cfg['model']['horizon']
    num_nodes  = cfg['model']['num_nodes']
    input_dim  = cfg['model']['input_dim']
    output_dim = cfg['model']['output_dim']

    y_preds  = []
    y_truths = []

    with torch.no_grad():
        for x_np, y_np in data['test_loader'].get_iterator():
            x = prepare_x(x_np, seq_len, num_nodes, input_dim).to(device)
            y = prepare_y(y_np, horizon, num_nodes, output_dim)
            output = model(x).cpu()    # (horizon, batch, num_nodes)
            y_preds.append(output)
            y_truths.append(y)

    y_preds  = torch.cat(y_preds,  dim=1)   # (12, N, 207)
    y_truths = torch.cat(y_truths, dim=1)

    per_horizon = {}
    for h in range(12):
        pred = scaler.inverse_transform(y_preds[h])
        real = scaler.inverse_transform(y_truths[h])
        mae, mape, rmse = metric(pred, real)
        per_horizon[h + 1] = dict(mae=mae, mape=mape * 100, rmse=rmse)
    return per_horizon


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()
    device = torch.device(args.device)
    rng = np.random.default_rng(args.seed)

    # Patch module-level device vars in DCRNN source so hidden states,
    # lazy weights, and sparse supports are all created on the right device.
    import model.pytorch.dcrnn_model as _dm
    import model.pytorch.dcrnn_cell as _dc
    _dm.device = device
    _dc.device = device

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    # Load raw adj for building variants
    _, _, raw_adj = dcrnn_utils.load_pickle(args.adjdata)
    raw_adj = np.array(raw_adj, dtype=np.float32)
    n = raw_adj.shape[0]

    density = (raw_adj > 0).mean()
    rand_adj = (rng.random((n, n)) < density).astype(np.float64)
    np.fill_diagonal(rand_adj, 1.0)
    perm = rng.permutation(n)
    perm_adj = raw_adj[np.ix_(perm, perm)]

    variants = {
        'original':              raw_adj,
        'zeros':                 np.zeros((n, n)),
        'identity':              np.eye(n),
        'random (same density)': rand_adj,
        'permuted nodes':        perm_adj,
        'fully connected':       np.ones((n, n)),
    }

    # Build data loader (uses normalized y — DCRNN normalizes both x and y)
    data = dcrnn_utils.load_dataset(**cfg['data'])
    scaler = data['scaler']

    # Build model with original adj and lazy-init weights via one forward pass
    import logging
    logger = logging.getLogger('dcrnn')
    model_kwargs = cfg['model']
    model = DCRNNModel(raw_adj, logger=logger, **model_kwargs)
    model.to(device)   # move projection_layer and buffers to device before lazy init
    model.eval()

    # Warm up: one val batch triggers lazy weight registration on the target device
    with torch.no_grad():
        for x_np, y_np in data['val_loader'].get_iterator():
            x = prepare_x(x_np, model_kwargs['seq_len'], n, model_kwargs['input_dim']).to(device)
            model(x)
            break

    # Now load checkpoint (weights exist, shapes match)
    checkpoint = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    print(f'Loaded: {args.checkpoint}')

    # Evaluate each variant
    all_results = {}
    for name, adj in variants.items():
        print(f'  evaluating: {name}...')
        supports = make_dcrnn_supports(adj)
        all_results[name] = run_inference(model, data, scaler, supports, cfg, device)

    # Print table
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
    print('Note: GRU weight matrices stay frozen -- deltas show diffusion graph contribution.')
    print('      Unlike GWN/DGCRN, DCRNN has NO adaptive adj -- full degradation is visible here.')

    from perturb import evaluate_operational
    eval_fn = lambda dense: run_inference(model, data, scaler, make_dcrnn_supports(dense), cfg, device)
    evaluate_operational(eval_fn, raw_adj, baseline, model_name='DCRNN',
                         modes=('edge_removal', 'weight_noise', 'targeted_betweenness'),
                         rates=(0.1, 0.2, 0.3),
                         csv_path='results_dcrnn_operational.csv')


if __name__ == '__main__':
    main()
