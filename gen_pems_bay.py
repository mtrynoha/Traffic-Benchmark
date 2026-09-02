import numpy as np
import pandas as pd

df = pd.read_hdf('data/pems-bay.h5')
print('Shape:', df.shape)

x_offsets = np.sort(np.concatenate((np.arange(-(12 - 1), 1, 1),)))
y_offsets = np.sort(np.arange(1, 13, 1))

num_samples, num_nodes = df.shape
data = np.expand_dims(df.values, axis=-1)
time_ind = (df.index.values - df.index.values.astype('datetime64[D]')) / np.timedelta64(1, 'D')
time_in_day = np.tile(time_ind, [1, num_nodes, 1]).transpose((2, 1, 0))
data = np.concatenate([data, time_in_day], axis=-1)

x, y = [], []
min_t = abs(min(x_offsets))
max_t = abs(num_samples - abs(max(y_offsets)))
for t in range(min_t, max_t):
    x.append(data[t + x_offsets, ...])
    y.append(data[t + y_offsets, ...])
x = np.stack(x, axis=0)
y = np.stack(y, axis=0)
print('x shape:', x.shape, ', y shape:', y.shape)

num_test = round(x.shape[0] * 0.2)
num_train = round(x.shape[0] * 0.7)
num_val = x.shape[0] - num_test - num_train

for cat, (_x, _y) in [('train', (x[:num_train], y[:num_train])),
                       ('val',   (x[num_train:num_train + num_val], y[num_train:num_train + num_val])),
                       ('test',  (x[-num_test:], y[-num_test:]))]:
    print(cat, 'x:', _x.shape, 'y:', _y.shape)
    np.savez_compressed(
        'data/PEMS-BAY/{}.npz'.format(cat),
        x=_x, y=_y,
        x_offsets=x_offsets.reshape(list(x_offsets.shape) + [1]),
        y_offsets=y_offsets.reshape(list(y_offsets.shape) + [1]),
    )

print('Done.')
