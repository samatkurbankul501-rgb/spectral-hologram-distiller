"""Print the current bench_cmapss_results.json medians compactly."""
import json

with open('bench_cmapss_results.json', encoding='utf-8') as f:
    d = json.load(f)
res = d['results']['fd001']
print('config:', d.get('config'))
def _r(x):
    return {k: (round(v, 3) if isinstance(v, (int, float)) else v) for k, v in x.items()}
print('oracle:', _r(res['oracle']))
print('partial:', res.get('partial'), '| n_seeds:', len(res['records']))
print()
print('wilcoxon (shd arms vs baselines, krr_crit):')
for arm in [k for k in res['wilcoxon'].keys() if k.endswith('_krr_rmse_crit')]:
    t = res['wilcoxon'][arm]
    print(f"  {arm}: {t['median_shd']:.3f} vs {t['median_' + arm.split('_vs_')[1].split('_')[0]]:.3f} "
          f"wins {t['wins_shd']}/{len(res['records'])} p={t['p_value']:.1e}")
print()
print('vs_oracle:')
for arm, vo in res.get('vs_oracle', {}).items():
    for m, v in vo.items():
        print(f"  {arm} {m}: median_shd={v['median_shd']:.3f} oracle={v['oracle']:.3f} "
              f"wins {v['wins_vs_oracle']}/{len(res['records'])}")
