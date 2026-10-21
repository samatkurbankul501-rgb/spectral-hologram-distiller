import json, sys, numpy as np

p = sys.argv[1]
d = json.load(open(p, encoding='utf-8'))
for key, r in d.get('results', {}).items():
    print('===', key, r.get('title'), 'n_seeds', r.get('n_seeds'), 'partial', r.get('partial'))
    print('oracle', {k: round(v, 4) for k, v in r.get('oracle', {}).items()})
    recs = r.get('records', [])
    if not recs:
        continue
    for arm in ['shd', 'random', 'kmeans', 'kt', 'skt']:
        if arm not in recs[0]:
            continue
        for met in ['krr_rmse_all', 'krr_rmse_crit', 'krr_r2', 'gbm_rmse_all', 'gbm_rmse_crit', 'rare_captured']:
            if met not in recs[0][arm]:
                continue
            vals = [rec[arm][met] for rec in recs]
            print(f'  {arm:7s} {met:14s} med={np.median(vals):.4f} iqr={np.percentile(vals,75)-np.percentile(vals,25):.4f}')
    print('  -- wilcoxon sample --')
    for k in list(r.get('wilcoxon', {}))[:6]:
        t = r['wilcoxon'][k]
        print('  ', k, {kk: (round(v, 6) if isinstance(v, float) else v) for kk, v in t.items()})
    print()
