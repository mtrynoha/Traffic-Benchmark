import sys, os
sys.path.insert(0, 'methods/DCRNN')
import yaml
from lib import utils as dcrnn_utils
from model.pytorch.dcrnn_model import DCRNNModel
from model.pytorch.dcrnn_cell import DCGRUCell
import numpy as np, torch

with open('methods/DCRNN/data/model/dcrnn_la.yaml') as f:
    cfg = yaml.safe_load(f)

_, _, raw_adj = dcrnn_utils.load_pickle('data/sensor_graph/adj_mx.pkl')
raw_adj = np.array(raw_adj, dtype=np.float32)

import logging
logger = logging.getLogger('dcrnn')
model = DCRNNModel(raw_adj, logger=logger, **cfg['model'])

data = dcrnn_utils.load_dataset(**cfg['data'])

model.eval()
with torch.no_grad():
    for x_np, y_np in data['val_loader'].get_iterator():
        x = torch.from_numpy(x_np).float().permute(1, 0, 2, 3).reshape(12, x_np.shape[0], 207 * 2)
        out = model(x)
        print('warm-up forward shape:', out.shape)
        break

chk = torch.load('models/la_best.tar', map_location='cpu')
model.load_state_dict(chk['model_state_dict'])
print('checkpoint loaded OK')
print('num encoder cells:', len(model.encoder_model.dcgru_layers))
print('num decoder cells:', len(model.decoder_model.dcgru_layers))
print('supports per cell:', len(model.encoder_model.dcgru_layers[0]._supports))
