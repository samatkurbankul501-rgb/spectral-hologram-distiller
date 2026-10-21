import json, sys
from collections import Counter

nb = json.load(open('Untitled22.ipynb', encoding='utf-8'))
cells = nb['cells']
print('nbformat:', nb.get('nbformat'), nb.get('nbformat_minor'))
print('kernel:', nb['metadata'].get('kernelspec', {}).get('name'))
print('total cells:', len(cells))
print('cell types:', dict(Counter(c['cell_type'] for c in cells)))
print()

for i, c in enumerate(cells):
    src = ''.join(c.get('source', []))
    lines = src.strip().splitlines()
    first = lines[0][:110] if lines else '(empty)'
    n_out = len(c.get('outputs', [])) if c['cell_type'] == 'code' else 0
    ex = c.get('execution_count')
    print(f"{i:3d} [{c['cell_type']:8s}] exec={ex} out={n_out:2d} len={len(src):6d} | {first}")
