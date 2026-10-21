import json, numpy as np

d = json.load(open(r'_release\spectral-hologram-distiller\bench_seeds_results.json', encoding='utf-8'))
for key in ['superconduct', 'reactor']:
    r = d['results'][key]
    print('###', key, 'n_train', r.get('n_train'), 'n_test', r.get('n_test'), 'n_test_crit', r.get('n_test_crit'))
    print('oracle', {k: round(v, 4) for k, v in r['oracle'].items()})
    recs = r['records']
    for arm in ['random', 'kmeans', 'kt', 'skt', 'shd']:
        vals = {}
        for met in ['krr_rmse_crit', 'krr_rmse_all', 'krr_r2', 'gbm_rmse_crit', 'gbm_rmse_all', 'rare_captured']:
            if met in recs[0][arm]:
                a = np.array([x[arm][met] for x in recs], float)
                vals[met] = (round(float(np.median(a)), 3), round(float(np.percentile(a, 75) - np.percentile(a, 25)), 3))
        print(f'{arm:7s}', vals)
    print('wilcoxon (wins_shd / p):')
    for k, v in r['wilcoxon'].items():
        if k.endswith('krr_rmse_crit') or k.endswith('krr_rmse_all') or k.endswith('gbm_rmse_all') or k.endswith('rare_captured'):
            print(' ', k, v.get('wins_shd'), v.get('p_value'), 'd=', v.get('median_shd'), v.get('median_kmeans') or v.get('median_random') or '')
    print()
