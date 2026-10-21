# Migrated into the repository on 2026-10-21: the absolute workspace
# paths below were replaced by self-locating resolvers so the script
# keeps working from any directory and from any clone.
from pathlib import Path as _Path
REPO_PATH = _Path(__file__).resolve().parents[2]
WORKSPACE_PATH = REPO_PATH.parents[1]
import re
from pathlib import Path

path = REPO_PATH / "paper" / "main.tex"
text = path.read_text(encoding="utf-8")
original = text

def sub(old, new, count=1):
    global text
    n = text.count(old)
    assert n == count, f"expected {count} occurrences, found {n}: {old[:80]!r}"
    text = text.replace(old, new, count)

# ---- Abstract: full block replacement (the line is too long to match verbatim)
new_abstract = r"""\begin{abstract}
Machine learning on high-throughput sensor telemetry and experimental physics streams faces a triad of unresolved challenges: (i) sample-efficiency bottlenecks where critical regimes can be a small fraction of observations (3\% in the reactor and PM2.5 streams; 18.6\% for the $T_c \ge 77\,\mathrm{K}$ superconductivity regime) and are under-represented by standard downsampling; (ii) prohibitively expensive dataset distillation methods that require hundreds of GPU-hours of bi-level optimization; and (iii) the inability of static coresets to adapt to non-stationary drift without re-processing the historical corpus. We propose \textbf{Spectral Hologram Distiller (SHD)}, a zero-GPU, bounded-memory streaming framework that compresses telemetry into compact ``crystals'' of real medoids with an analytical spectral passport. SHD projects data into a dual-scale Bochner random Fourier feature space, triggers candidates via a two-tailed residual-leverage filter, and clusters the coupled feature-target phase manifold $[\mathbf{Z} \parallel \kappa \cdot y]$ to allocate dedicated Voronoi cells to rare regimes. Coreset weights are solved at crystallization via non-negative FISTA herding and blended with tempered Voronoi volumes. Each crystal stores a 16-bit top-32 spectral passport enabling a millisecond-scale Bures--Wasserstein radar (4--84\,ms depending on window size) that flags out-of-distribution drift without raw data, followed by surgical healing with bounded forgetting. On the 81-dimensional UCI Superconductivity benchmark across 20 seeds, a 300-medoid crystal ($\approx 98$\,KB, $70\times$ compression) reaches a critical-regime ($T_c \ge 77\,\mathrm{K}$) KRR RMSE of $18.81\,\mathrm{K}$, outperforming Random ($22.43\,\mathrm{K}$), $K$-Means ($22.93\,\mathrm{K}$), Kernel Thinning ($23.26\,\mathrm{K}$), and Supervised Kernel Thinning ($21.20\,\mathrm{K}$; all $p = 1.9 \times 10^{-6}$, Holm $4.6 \times 10^{-5}$), while approaching the 9,000-point KRR oracle ($17.59\,\mathrm{K}$; $+6.9\%$). In Bayesian optimization for phase-boundary recovery, SHD finds 6 of 12 pocket points in its top-20 EI acquisitions versus 0 for both baselines, with F1 $= 1.000$.
\end{abstract}"""
text, n = re.subn(r"\\begin\{abstract\}.*?\\end\{abstract\}", lambda m: new_abstract, text, count=1, flags=re.S)
assert n == 1, "abstract block not found"

# ---- Introduction / contributions
sub(
    r"over 97\% of samples reflect stable, uninformative background equilibria, while the critical phenomena---exothermic runaway, structural shear bursts, or high-temperature superconducting phases---occur in narrow, transient pockets of the parameter space.",
    r"in the reactor and PM2.5 streams over 97\% of samples reflect stable, uninformative background equilibria, while the critical phenomena---exothermic runaway, structural shear bursts, or high-temperature superconducting phases---occur in narrow, transient pockets; in superconductivity the critical $T_c \ge 77\,\mathrm{K}$ regime is 18.6\% of the data.",
)
sub(
    r"In a 300-point coreset from 20,000 observations, a 2\% critical regime receives 4--6 points, which is insufficient to resolve local gradient curvature for surrogate modeling.",
    r"A 2\% critical regime receives roughly its proportional share (about 6 of 300 points), which is often insufficient to resolve local gradient curvature for surrogate modeling; KT captures 59/300 critical superconductors (19.7\%) and 10/300 reactor peaks (3.3\%), i.e.\ proportional rather than dedicated representation.",
)
sub(
    r"This guarantees that rare regimes occupy dedicated Voronoi cells and cannot be pruned by dense background clusters.",
    r"This allocates dedicated Voronoi cells to rare regimes; the band is protected during pruning, although a tight memory cap can still remove non-band medoids.",
)
sub(
    r"By projecting incoming windows onto this 32-dimensional subspace, a Bures--Wasserstein metric \citep{bhatia2019bures} detects out-of-distribution drift in under 4 milliseconds on CPU without raw data access.",
    r"By projecting incoming windows onto this 32-dimensional subspace, a Bures--Wasserstein metric \citep{bhatia2019bures} detects out-of-distribution drift in 4--84\,ms on CPU (window-size dependent) without raw data access.",
)
sub(
    r"This eliminates catastrophic forgetting without storing past streams.",
    r"This limits catastrophic forgetting without storing past streams: old-domain RMSE degrades from 7.32 to 13.36\,K, versus 20.62\,K under naive overwrite.",
)
sub(
    r"Across 20 independent seeds on the 81-dimensional UCI Superconductivity dataset \citep{hamidieh2018data} and a 4D phase-transition reactor, SHD demonstrates statistically significant superiority ($p < 10^{-5}$, paired Wilcoxon) over Random, $K$-Means, Kernel Thinning \citep{dwivedi2024kernel}, and Supervised Kernel Thinning \citep{gong2024supervised}.",
    r"Across 20 independent seeds on the 81-dimensional UCI Superconductivity dataset \citep{hamidieh2018data} and a 4D phase-transition reactor, SHD demonstrates statistically significant superiority on the critical regime ($p < 10^{-5}$, paired Wilcoxon, Holm-corrected) over Random, $K$-Means, Kernel Thinning \citep{dwivedi2024kernel}, and Supervised Kernel Thinning \citep{gong2024supervised}; global metrics are competitive but not uniformly superior (SKT wins SC GBM-all by 0.09\,K, n.s.; $K$-Means wins reactor GBM-all by 0.03\,K, n.s.).",
)

# ---- Method section
sub(
    r"SHD first normalizes features online using streaming statistics $\boldsymbol{\mu}_X, \boldsymbol{\sigma}_X$.",
    r"SHD first normalizes features with statistics fixed at the first batch (target statistics can optionally adapt); the warm-up buffer is capped at 8,192 rows.",
)
sub(
    r"During initial warm-up, points are partitioned into background ($1 - \beta = 74\%$) and phase-peak ($\beta = 26\%$) subsets.",
    r"During initial warm-up, the medoid budget is split into background ($1 - \beta = 74\%$) and phase-peak ($\beta = 26\%$) subsets; the 84th-percentile trigger typically flags about 16\% of rows.",
)
sub(
    r"points falling into cell $j$ update centroid locations with an adaptive learning rate $\eta = \min(0.25, B / n_{\text{seen}})$. If a candidate row within cell $j$ demonstrates higher phase score and lower quantization error than the active medoid, it replaces the active medoid.",
    r"points falling into cell $j$ update centroid locations with an adaptive learning rate $\eta = \min(0.25, n_{\text{new}}/n_{\text{cell}})$. If a candidate row within cell $j$ has lower quantization error to the centroid than the active medoid, it replaces it; the phase score is used for band allocation, not for medoid replacement.",
)
sub(
    r"Rather than assigning uniform weights $1/M$, SHD solves for empirical coreset weights $\boldsymbol{\gamma}^* \in \mathbb{R}_+^M$",
    r"Rather than assigning uniform weights $1/M$, SHD solves at crystallization time for empirical coreset weights $\boldsymbol{\gamma}^* \in \mathbb{R}_+^M$",
)
sub(
    r"with herding weights:",
    r"with herding weights (both terms normalized to mean one):",
)
sub(
    r"\textbf{Spectral Passport:} Upon crystallization, we compute the top-$r$ ($r=32$) eigenpairs of the whitened feature covariance:",
    r"\textbf{Spectral Passport:} Upon crystallization, we compute the top-$r$ ($r=32$) eigenpairs of the feature covariance:",
)
sub(
    r"The resulting artifact ($\approx 98\,\mathrm{KB}$) stores",
    r"The resulting artifact ($\approx 98\,\mathrm{KB}$ for the 300-medoid superconductivity crystal; $\approx 108\,\mathrm{KB}$ with the stored projector) stores",
)
sub(
    r"Because operations are performed on $32 \times 32$ matrices, detection latency is under $4\,\mathrm{ms}$.",
    r"Because operations are performed on $32 \times 32$ matrices, detection latency is 4--84\,ms depending on window size ($B = 64\ldots 5{,}780$) on this CPU, not sub-millisecond.",
)
sub(
    r"\item Sample $n_{\text{inject}}$ new candidate medoids from $\mathbf{X}_{\text{new}}$ via phase-weighted $k$-means++.",
    r"\item Sample $n_{\text{inject}}$ new candidate medoids from $\mathbf{X}_{\text{new}}$ via $k$-means++ in the frozen RFF space.",
)
sub(
    r"\item Append injected medoids to the crystal. If crystal size exceeds $M_{\max}$, prune only from the lowest-weight background medoids while strictly protecting the top 26\% phase-peak medoids.",
    r"\item Append injected medoids to the crystal. If crystal size exceeds $M_{\max}$, prune the lowest-weight non-protected medoids, protecting the top 26\% phase-peak medoids and all injected points; the cap is hard and raises if it cannot accommodate them.",
)
sub(
    r"\item Blend the spectral passport: $\boldsymbol{\Lambda} \leftarrow (1-\alpha) \boldsymbol{\Lambda} + \alpha \text{diag}(\mathbf{C}_{\text{new}})$, with $\alpha = \text{clip}(B / (n_{\text{seen}} + B), 0.15, 0.50)$.",
    r"\item Blend the spectral passport: $\boldsymbol{\Psi} \leftarrow (1-\alpha)\boldsymbol{\Psi} + \alpha \boldsymbol{\Psi}_{\text{new}}$, then re-eigendecompose and store the top-32 eigenpairs; $\alpha = \text{clip}(B / (n_{\text{seen}} + B), 0.15, 0.50)$.",
)

# ---- Experimental setup
sub(
    r"We evaluate SHD across 20 independent seeds on two rigorous testbeds:",
    r"We evaluate SHD across 20 independent seeds on two testbeds (one real, one synthetic):",
)
sub(
    r"comprising 8.7\% of the dataset.",
    r"comprising 18.6\% of the dataset (3,952 / 21,263; 778 of 4,253 test rows).",
)
sub(
    r"All tests report median and interquartile range (IQR) across 20 seeds; significance is assessed via two-sided paired Wilcoxon signed-rank tests against SHD.",
    r"All tests report median and interquartile range (IQR) across 20 seeds; significance is assessed via two-sided paired Wilcoxon signed-rank tests against SHD, Holm-adjusted within each dataset/metric family. Tables report raw $p$; the text notes where Holm changes a conclusion. The Wins column counts seeds in which SHD has lower critical-regime KRR RMSE.",
)

# ---- Oracle labels and wins column
sub(r"Oracle (9,000 pts)", r"Oracle (KRR 9,000; GBM full train)")
sub(r"Oracle (16,000 pts)", r"Oracle (KRR 9,000; GBM 16,000)")
sub(r"\textbf{Wins vs SHD}", r"\textbf{SHD wins (KRR Crit)}", count=2)
sub(r"& 0 / 20 ($p < 10^{-5}$) \\", r"& 20 / 20 ($p < 10^{-5}$) \\", count=4)
sub(r"& 0 / 20 ($p = 1.9 \times 10^{-6}$, KRR Crit) \\", r"& 20 / 20 ($p = 1.9 \times 10^{-6}$, KRR Crit) \\", count=2)

# ---- Main results paragraph + new controls subsection
old_start = "As detailed in Table~\\ref{tab:superconduct}"
lines = text.split("\n")
idx = next(i for i, line in enumerate(lines) if line.startswith(old_start))
new_results = r"""As detailed in Tables~\ref{tab:superconduct} and~\ref{tab:reactor}, SHD is best-in-class on the critical regime but not uniformly superior on global metrics. On Superconductivity, SHD reaches $18.81\,\mathrm{K}$ on the liquid-nitrogen regime, approaching the 9,000-point KRR oracle ($17.59\,\mathrm{K}$; $+6.9\%$) and outperforming SKT ($21.20\,\mathrm{K}$) by $2.4\,\mathrm{K}$ and KT ($23.26\,\mathrm{K}$) by $4.5\,\mathrm{K}$ across all 20 seeds ($20/20$, $p=1.9\times10^{-6}$; Holm-adjusted $4.6\times10^{-5}$). SKT remains the strongest baseline on global KRR ($15.78\,\mathrm{K}$; SHD wins $17/20$, $p=6.4\times10^{-3}$) and is numerically ahead on GBM-all ($15.97$ vs $16.06\,\mathrm{K}$, SHD wins $6/20$, n.s.\ $p=0.070$; Holm $p=0.278$). On the 4D reactor, SHD wins the critical regime $18/20$ against KT ($p=9.5\times10^{-6}$) and $20/20$ against SKT ($p=1.9\times10^{-6}$), $K$-Means and Random; $K$-Means remains better on GBM-all ($0.596$ vs $0.624$, SHD wins $8/20$, n.s.\ $p=0.245$). Per crystal, SHD retains $64.5$ critical superconductors versus $59.0$ (KT), $55.0$ (SKT), $57.0$ (Random), and $31.0$ ($K$-Means), and $19.5$ reactor peak points versus $10.0$ (KT), $8.5$ (SKT), $9.0$ (Random) and $11.5$ ($K$-Means). KT/SKT are not rare-collapsed: their captures are close to the population share ($19.7\%$ and $18.3\%$ on SC; $3.3\%$ and $2.8\%$ on the reactor), whereas SHD over-represents the band. The contribution is therefore over-representation and critical accuracy, not prevention of rare erasure.

\subsection{Data-Access, Sampling, and Weighting Controls}

A natural objection is that SHD sees the full stream while Kernel Thinning is run on a seeded pre-sample. We therefore repeat the comparison on the same 4{,}800-row pre-sample for both methods: SHD\_sub4800 reaches $18.71\,\mathrm{K}$ critical KRR RMSE versus $23.19\,\mathrm{K}$ for KT ($15/15$ paired splits, $p=6.1\times10^{-5}$) and $21.70\,\mathrm{K}$ for SKT ($15/15$, $p=6.1\times10^{-5}$). The advantage is not explained by data access. A second objection is that reserving cells by target value is merely stratified sampling. A tail-stratified baseline (26\% of the budget from the top 26\% of $y$, 74\% random background) reaches $24.09\,\mathrm{K}$ versus $19.20\,\mathrm{K}$ for full SHD on the same 15 splits ($15/15$, $p=6.1\times10^{-5}$); the phase-manifold geometry, not stratification alone, is doing the work.

Finally, we isolate the FISTA herding weights by evaluating the \emph{same} crystal twice, once with $\gamma$ and once with uniform weights (20 seeds; Table~\ref{tab:weights}). The weights are a global/critical trade-off, not a free win: on Superconductivity they improve KRR-all from $15.99$ to $15.57\,\mathrm{K}$ ($20/20$, $p=1.9\times10^{-6}$) while leaving critical KRR statistically unchanged ($18.98$ vs $19.38\,\mathrm{K}$, $11/20$, $p=0.57$); on the reactor they improve KRR-all ($0.975$ vs $0.992$, $20/20$, $p=1.9\times10^{-6}$) but \emph{cost} critical KRR ($3.426$ vs $3.335\,\mathrm{K}$; uniform wins $20/20$, $p=1.9\times10^{-6}$). The critical-regime headline is therefore attributable to medoid selection, not to the weighting; for a critical-only deployment, uniform weights are preferable on the reactor and neutral on Superconductivity.

\begin{table}[h]
\centering
\small
\caption{Weighting ablation on identical crystals (20 seeds, $M=300$). Lower is better; bold marks the better median.}
\label{tab:weights}
\begin{tabular}{lcccc}
\toprule
\textbf{Dataset} & \textbf{KRR All ($\gamma$ / unif.)} & \textbf{KRR Crit ($\gamma$ / unif.)} & \textbf{GBM Crit ($\gamma$ / unif.)} & \textbf{Winner (KRR Crit)} \\
\midrule
Superconductivity & \textbf{15.57} / 15.99 & \textbf{18.98} / 19.38 & \textbf{18.00} / 18.07 & tie (11/20, n.s.) \\
4D Reactor & \textbf{0.975} / 0.992 & 3.426 / \textbf{3.335} & \textbf{1.77} / 2.17 & uniform (20/20) \\
\bottomrule
\end{tabular}
\end{table}"""
lines[idx] = new_results
text = "\n".join(lines)

# ---- BO wording
sub(
    r"top-20 EI acquisitions land $6.0$ times inside the critical pocket versus $0.0$ for both baselines",
    r"top-20 EI acquisitions land $6.0$ times inside the 12-point critical pocket versus $0.0$ for both baselines",
)

# ---- Healing section
old_radar = r"    \item \textbf{Radar Performance:} The Bures--Wasserstein radar achieves $20/20$ shift detections with $0/20$ false positives on stationary streams. Median evaluation latency is $3.1\,\mathrm{ms}$."
new_radar = r"    \item \textbf{Radar Performance:} The Bures--Wasserstein radar achieves $20/20$ shift detections with $0/20$ false positives on stationary streams. Median latency for the 7,300-row Domain B window is $50.2\,\mathrm{ms}$ ($4$--$84\,\mathrm{ms}$ across window sizes); it is not sub-millisecond."
sub(old_radar, new_radar)
old_heal = r"    \item \textbf{Healing and Memory Retention:} Upon receiving $45$ injected points with total crystal size bounded at $M = 90$, Domain B RMSE drops from $57.17\,\mathrm{K}$ to $26.88\,\mathrm{K}$ ($p = 1.9 \times 10^{-6}$). Crucially, Domain A performance is strictly preserved: Domain A RMSE is $13.36\,\mathrm{K}$ under SHD healing versus $20.62\,\mathrm{K}$ when naive overwriting is applied ($p = 1.9 \times 10^{-6}$), demonstrating total mitigation of catastrophic forgetting."
new_heal = r"    \item \textbf{Healing and Memory Retention:} With $45$ injected points and a hard cap of $220$ medoids ($245$ unbounded), Domain B RMSE drops from $57.17\,\mathrm{K}$ to $26.88\,\mathrm{K}$ ($20/20$, $p = 1.9 \times 10^{-6}$). Old-domain performance is not preserved: Domain A degrades from $7.32\,\mathrm{K}$ (pre-heal) to $13.36\,\mathrm{K}$, but remains $7.26\,\mathrm{K}$ better than naive overwrite ($20.62\,\mathrm{K}$; $20/20$, $p = 1.9 \times 10^{-6}$). Healing is therefore a bounded trade-off, not zero forgetting. The post-heal BW distance is $0.465$, still above the $0.085$ alert threshold: one heal does not clear the alert; repeated heals or a domain bank are required."
sub(old_heal, new_heal)

# ---- Ablation reading
sub(
    r"removing the phase coordinate is the single largest degradation",
    r"removing the phase coordinate is a large degradation, but a narrow band is worse",
)
sub(
    "Second, the only constant whose removal is catastrophic is \\texttt{phase\\_kappa}:\nthe coupled feature--target manifold, not the weighting scheme, is the actual\ncontribution. We therefore recommend keeping $\\kappa > 0$ and treating the other\nconstants as tunable.",
    r"Second, \texttt{phase\_kappa} is the structural constant (removal costs $0.85\,\mathrm{K}$), but it is not the only sensitive one: \texttt{high\_band\_ratio}$=0.10$ costs $1.09\,\mathrm{K}$. We recommend keeping $\kappa>0$ and tuning the band width on validation data. The ablation uses 5 seeds; a 10-seed CI replication is reported in the repository.",
)

# ---- Limitations paragraph
old_tradeoff = r"\textbf{Trade-off Transparency:} In accordance with empirical honesty, SHD is an engineered trade-off: allocating dedicated Voronoi volume to rare regimes slightly degrades global background modeling ($0.546$ vs $0.425$ background RMSE in the BO benchmark). Furthermore, on smooth Gaussian data lacking distinct phase transitions, $K$-Means achieves comparable global representation. SHD is specifically designed for multi-scale physical streams where extreme operational modes carry asymmetric utility."
new_tradeoff = old_tradeoff + r"""

\textbf{Limitations.} (i) The reactor is synthetic; the only real regression dataset is Superconductivity. On a second real stream (Beijing PM2.5, 3\% critical), single-crystal SHD loses global RMSE ($95.1$ vs $89.9$ KT / $88.2$ Random) and only ties KT on critical ($231.0$ vs $245.7$; $13/20$, $p=0.216$); SHD-C plus a 300-medoid global guard recovers $196.0/85.3$ and ties KT on GBM-all ($78.9$), i.e.\ the fix requires a second crystal, not the single-crystal method. (ii) The radar separates strong shifts perfectly (AUC $=1.0$), but graded/noisy shifts are untested. (iii) One heal does not clear the alert ($BW=0.465 > 0.085$); a domain bank or repeated heals are needed. (iv) We provide no coverage guarantee; the method is empirical. (v) The 15-split data-access and stratification controls use a single dataset."""
sub(old_tradeoff, new_tradeoff)

path.write_text(text, encoding="utf-8")
print(f"written {len(text)} chars; delta {len(text) - len(original):+d}")
