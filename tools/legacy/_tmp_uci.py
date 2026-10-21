import numpy as np
try:
    from sklearn.datasets import fetch_openml
    d = fetch_openml('superconduct', version=1, as_frame=True, parser='auto')
    y = d.target.to_numpy(dtype=float)
    print('n', len(y), 'frac>=77', float(np.mean(y >= 77)), 'n>=77', int(np.sum(y >= 77)))
    print('min/max', y.min(), y.max())
except Exception as e:
    print('FETCH_FAILED', type(e).__name__, str(e)[:300])
