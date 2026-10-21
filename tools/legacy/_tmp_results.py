import json, sys, numpy as np

base = sys.argv[1]
for name in ['bench_heal_superconduct_results.json', 'bench_deltas_results.json',
             'bridge_neon_bo_results.json', 'bench_ablation_results.json']:
    p = base + '\\' + name
    print('#' * 20, name)
    d = json.load(open(p, encoding='utf-8'))
    if 'summary' in d:
        print(json.dumps(d['summary'], indent=1)[:3000])
    if 'wilcoxon' in d:
        w = d['wilcoxon']
        print('-- wilcoxon --')
        for k, v in w.items():
            print(k, json.dumps(v)[:500])
    if 'wilcoxon_deltas_on_vs_off' in d:
        print('-- deltas wilcoxon --')
        for k, v in d['wilcoxon_deltas_on_vs_off'].items():
            print(k, v)
    if 'config' in d:
        print('config', d['config'])
    print()
