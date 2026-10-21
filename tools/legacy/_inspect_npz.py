import numpy as np
import zipfile

p = '_release/spectral-hologram-distiller/artifacts/superconduct_v37.hologram.npz'
z = zipfile.ZipFile(p)
print('zip entries:', z.namelist())
print()
d = np.load(p, allow_pickle=True)
for k in d.files:
    arr = d[k]
    line = f'{k}: shape={arr.shape} dtype={arr.dtype}'
    if arr.dtype == object:
        v = arr.item()
        line += f' -> {type(v).__name__}'
        if isinstance(v, dict):
            print(line)
            for kk, vv in v.items():
                s = repr(vv)
                if len(s) > 260:
                    s = s[:260] + '...'
                print(f'    {kk}: {s}')
        else:
            print(line)
            print('  repr:', repr(v)[:800])
    else:
        if arr.size <= 16:
            line += f' values={arr.ravel()[:16]}'
        else:
            line += f' min={arr.min():.6g} max={arr.max():.6g}'
        print(line)
