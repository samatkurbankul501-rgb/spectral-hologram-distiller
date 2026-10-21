import re, textwrap
s = open(r'_release\spectral-hologram-distiller\paper\main.tex', encoding='utf-8').read()
a = re.search(r'\\begin\{abstract\}(.*?)\\end\{abstract\}', s, re.S).group(1)
print(textwrap.fill(a.strip(), 120))
