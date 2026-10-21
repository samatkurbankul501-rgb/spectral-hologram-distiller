import json
p = r'_release\spectral-hologram-distiller\bench_heal_superconduct_results.json'
d = json.load(open(p, encoding='utf-8'))
print(json.dumps(d.get('summary', d), indent=1)[:4000])
print('--- config ---')
print(json.dumps(d.get('config', {}), indent=1)[:1000])
print('--- wilcoxon ---')
print(json.dumps(d.get('wilcoxon', {}), indent=1)[:2000])
