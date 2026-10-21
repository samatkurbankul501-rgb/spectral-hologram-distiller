# Migrated into the repository on 2026-10-21: the absolute workspace
# paths below were replaced by self-locating resolvers so the script
# keeps working from any directory and from any clone.
from pathlib import Path as _Path
REPO_PATH = _Path(__file__).resolve().parents[2]
WORKSPACE_PATH = REPO_PATH.parents[1]
from pathlib import Path
p = REPO_PATH / "paper" / "main.tex"
t = p.read_text(encoding="utf-8")

section = r"""\subsection{Second Real Stream: Beijing PM2.5}
\label{sec:pm25}

The superconductivity benchmark has an 18.6\% critical regime; the reactor is synthetic. We therefore evaluate the UCI Beijing PM2.5 stream \citep{chen2015pm25} (41,757 rows, 14 features, time-ordered 80/20 split; critical $=324\,\mu g/m^3$, the top 3\%, with 374 critical test rows). This is a deliberately harder regime for the default 26\% band: only 3\% of rows are critical, and the stream has strong seasonal drift.

Table~\ref{tab:pm25} shows the honest result. The single-crystal default loses global RMSE to KT ($95.1$ vs $89.9\,\mu g/m^3$) and Random ($88.2$), and its critical advantage over KT is not significant ($231.0$ vs $245.7$; 13/20, $p=0.216$). The critical-aware band (SHD-C, $\beta$ set from the observed critical fraction) improves critical RMSE to $216.6$ and rare capture to 25, but degrades global KRR to $101.2$. Adding a 300-medoid global guard recovers the global stream ($85.3$ KRR-all, $78.9$ GBM-all) while keeping critical RMSE at $196.0$ and rare capture at 29. This confirms that SHD is not a universal single-crystal replacement: for 3\% regimes the default band must be critical-aware, and production deployments need either a guard or a domain bank (Section~\ref{sec:passport}).

\begin{table}[t]
\centering
\small
\caption{Beijing PM2.5 (14 features, time-ordered split, critical $=324\,\mu g/m^3$, 374 critical test rows). RMSE in $\mu g/m^3$; lower is better. SHD and baselines use 20 seeds; SHD-C uses 20 seeds; the SHD-C+guard row uses 5 seeds.}
\label{tab:pm25}
\begin{tabular}{lccccc}
\toprule
\textbf{Method} & \textbf{KRR All} & \textbf{KRR Crit} & \textbf{GBM All} & \textbf{GBM Crit} & \textbf{Rare} \\
\midrule
Random & 88.24 & 248.92 & 80.05 & 225.51 & 9 \\
$K$-Means & 88.92 & 241.18 & \textbf{78.31} & 213.79 & 6 \\
Kernel Thinning & 89.89 & 245.74 & 79.85 & 228.81 & 9 \\
Supervised KT & 86.53 & 246.02 & 78.87 & 221.19 & 9 \\
SHD (default 26\% band) & 95.08 & 231.01 & 86.11 & 211.91 & 10 \\
SHD-C (critical-aware band) & 101.15 & 216.57 & 91.26 & 198.42 & 25 \\
SHD-C + 300-medoid guard & \textbf{85.32} & \textbf{196.02} & 78.92 & \textbf{168.40} & \textbf{29} \\
\bottomrule
\end{tabular}
\end{table}

"""

marker = r"\subsection{Bayesian Optimization and Critical Boundary Discovery}"
assert t.count(marker) == 1
t = t.replace(marker, section + marker, 1)

old_setup = r"We evaluate SHD across 20 independent seeds on two testbeds (one real, one synthetic):"
new_setup = r"We evaluate SHD across 20 independent seeds on two primary testbeds (one real, one synthetic), plus a third real stream in Section~\ref{sec:pm25}:"
assert t.count(old_setup) == 1
t = t.replace(old_setup, new_setup)

old_lim = r"(i) The reactor is synthetic; the only real regression dataset is Superconductivity. On a second real stream (Beijing PM2.5, 3\% critical), single-crystal SHD loses global RMSE ($95.1$ vs $89.9$ KT / $88.2$ Random) and only ties KT on critical ($231.0$ vs $245.7$; $13/20$, $p=0.216$); SHD-C plus a 300-medoid global guard recovers $196.0/85.3$ and ties KT on GBM-all ($78.9$), i.e.\ the fix requires a second crystal, not the single-crystal method."
new_lim = r"(i) The reactor is synthetic; the second real stream (Beijing PM2.5, 3\% critical, Section~\ref{sec:pm25}) shows that the single-crystal default loses global RMSE, and the fix requires a critical-aware band plus a guard or bank rather than the single-crystal method."
assert t.count(old_lim) == 1
t = t.replace(old_lim, new_lim)

p.write_text(t, encoding="utf-8")
print("ok")