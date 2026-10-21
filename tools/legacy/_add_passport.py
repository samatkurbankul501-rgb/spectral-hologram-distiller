# Migrated into the repository on 2026-10-21: the absolute workspace
# paths below were replaced by self-locating resolvers so the script
# keeps working from any directory and from any clone.
from pathlib import Path as _Path
REPO_PATH = _Path(__file__).resolve().parents[2]
WORKSPACE_PATH = REPO_PATH.parents[1]
from pathlib import Path
p = REPO_PATH / "paper" / "main.tex"
t = p.read_text(encoding="utf-8")

section = r"""\subsection{Passport Recalibration and Domain Banks}
\label{sec:passport}

A single blended passport cannot represent two strongly separated regimes. Across 5 seeds on the superconductivity A/B split ($M=200$, 45 injected points), the pre-heal radar accepts A ($BW=0.035$) and flags B ($0.629$). After the default blend, $BW_A=0.203$ and $BW_B=0.458$: \emph{both} domains exceed the 0.085 threshold, so the radar has no calibrated in-distribution domain. Setting \texttt{passport\_recalibrate=True} replaces the passport with the new window's covariance: B is accepted ($0.038$) and A is flagged ($0.497$). A \texttt{DomainBank} with one crystal/passport per regime routes A and B correctly $5/5$ and accepts both $5/5$; it also avoids forgetting by construction. We therefore recommend recalibration only when the old domain will not return, and a domain bank when both regimes remain live.

\begin{table}[t]
\centering
\small
\caption{Passport modes after an A$\to$B heal (superconductivity, 5 seeds, $M=200$; threshold $0.085$). Lower BW is better; ``in'' means below threshold. The bank routes correctly $5/5$ and accepts both domains $5/5$.}
\label{tab:passport}
\begin{tabular}{lccc}
\toprule
\textbf{Mode} & \textbf{BW on A} & \textbf{BW on B} & \textbf{Decision} \\
\midrule
Pre-heal & 0.035 & 0.629 & A in, B out \\
Blend (default) & 0.203 & 0.458 & both out \\
Recalibrate & 0.497 & 0.038 & A out, B in \\
Domain bank (nearest) & below threshold & below threshold & both in \\
\bottomrule
\end{tabular}
\end{table}

"""

marker = r"\subsection{Hyperparameter Sensitivity and Ablation}"
assert t.count(marker) == 1
t = t.replace(marker, section + marker, 1)

old = r"The post-heal BW distance is $0.465$, still above the $0.085$ alert threshold: one heal does not clear the alert; repeated heals or a domain bank are required."
new = r"The post-heal BW distance is $0.465$, still above the $0.085$ alert threshold; Section~\ref{sec:passport} shows that the default blend can leave \emph{both} domains OOD, that \texttt{passport\_recalibrate=True} accepts the new domain while flagging the old one, and that a \texttt{DomainBank} accepts both."
assert t.count(old) == 1
t = t.replace(old, new)

old_lim = r"(iii) One heal does not clear the alert ($BW=0.465 > 0.085$); a domain bank or repeated heals are needed."
new_lim = r"(iii) The default blended passport can leave both domains above threshold; recalibration accepts the new domain but flags the old, and a domain bank accepts both (Section~\ref{sec:passport})."
assert t.count(old_lim) == 1
t = t.replace(old_lim, new_lim)

old_pm = r"critical $=324\,\mu g/m^3$, the top 3\%, with 374 critical test rows"
new_pm = r"critical $=324\,\mu g/m^3$ (top 3\% of the training stream), with 374 critical test rows"
assert t.count(old_pm) == 1
t = t.replace(old_pm, new_pm)

old_tab = r"\bottomrule" + "\n" + r"\end{tabular}" + "\n" + r"\end{table}"
# Target the PM2.5 table by its unique preceding caption; replace only the first occurrence after it.
pm_marker = r"\label{tab:pm25}"
idx = t.index(pm_marker)
tail = t[idx:]
assert tail.count(old_tab) >= 1
new_tab = r"\bottomrule" + "\n" + r"\end{tabular}" + "\n" + r"\vspace{2pt}\footnotesize{The guard row is paired with a 5-seed SHD-C baseline (207.98 / 98.75 / 193.70 / 87.85; rare 25); the 20-seed SHD-C row is shown for reference.}" + "\n" + r"\end{table}"
tail = tail.replace(old_tab, new_tab, 1)
t = t[:idx] + tail

p.write_text(t, encoding="utf-8")
print("ok")