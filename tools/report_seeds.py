"""Print the current bench_seeds_results.json medians compactly."""
import json

with open('bench_seeds_results.json', encoding='utf-8') as f:
    d = json.load(f)
print('config:', d.get('config'))
for k, r in d['results'].items():
    print()
    print('===', k, r['title'], '| seeds:', r['n_seeds'], '| partial:', r.get('partial'))
    print('oracle:', {kk: round(vv, 3) for kk, vv in r['oracle'].items()})
    for m in ['krr_rmse_all', 'krr_rmse_crit', 'gbm_rmse_all', 'gbm_rmse_crit']:
        line = f'  {m:15s}'
        for p in ['random', 'kmeans', 'kt', 'skt']:
            t = r['wilcoxon'].get(f'shd_vs_{p}_{m}')
            if t:
                line += (f' | vs {p}: {t["median_shd"]:.3f}/{t["median_" + p]:.3f}'
                         f' wins {t["wins_shd"]}/{r["n_seeds"]} p={t["p_value"]:.1e}')
        print(line)
