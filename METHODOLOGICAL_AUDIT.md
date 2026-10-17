# SHD: методологический аудит (2026-10-17)

Аудит десяти дыр, которые рецензент нашёл бы в v3.9.1. Все десять закрыты:
девять экспериментами (включая второй датасет), один вендорингом кода.
Главный вывод: **ядро SHD выжило, три заявки paper — нет.**

## TL;DR

- **KT subsample confound закрыт:** SHD на тех же 4 800 точках, что и KT,
  даёт **18.71 K** против **23.19 K** у KT (15/15, p = 6.1e-5). Победа не
  объясняется доступом к данным.
- **Tail-stratified baseline не закрывает разрыв:** SHD **19.20 K** против
  **24.09 K** (15/15, p = 6.1e-5).
- **3 split × 5 seeds:** SHD выигрывает все 15/15 у KT/SKT/stratified на
  критическом режиме; Holm-adjusted p = 0.0026.
- **Lifelong heal учит новые домены, но забывает старые:** после 3-shift
  D 80.6 → 22.0, C 47.0 → 16.3, B 18.5 → 9.1, но A 3.89 → 12.49 (×3.2);
  BW остаётся ~0.51. «Zero catastrophic forgetting» неверно; нужен
  retention-механизм.
- **Residual trigger не помогает редкому режиму:** выключение residual
  улучшает critical RMSE на SC на 0.80 K (CI [-1.30, -0.42]) и ухудшает
  global RMSE на 0.39 K (CI [0.04, 0.48]). Это trade-off, не rare-regime win.
- **Dual-scale RFF не подтверждён:** single-scale 3.21 vs dual 3.44 на
  reactor, но CI разницы [-0.40, +0.03] включает 0.
- **Второй реальный датасет (Beijing PM2.5):** отрицательный результат.
  Critical 231.0 vs KT 245.7 (13/20, p = 0.216), но global KRR/GBM значимо
  хуже (wins 0–4/20); дефолтная 26% полоса не подходит для 3% режима.
- **ROC/AUC радара:** AUC = 1.0 на обоих сдвигах и всех полосах, но задача
  слишком лёгкая; это характеристика, а не доказательство graceful degradation.
- **Holm:** главный critical-claim выживает (p_holm = 4.6e-5); KRR-all vs SKT
  становится пограничным (p_holm = 0.051).
- **Ablation CI:** `phase_kappa=0` — **не** «single largest degradation»:
  `high_band_ratio=0.10` хуже (4.249 vs 4.132; diff +0.752 vs +0.668).

## Таблица: что закрыто и с каким результатом

| # | Дыра | Эксперимент | Результат | Вердикт |
| :--- | :--- | :--- | :--- | :--- |
| 1 | KT subsample confound | `bench_gaps.py`: SHD на тех же 4 800 | SHD_sub **18.71**, KT **23.19**, SKT **21.70**; 15/15 vs KT/SKT, p = 6.1e-5 | **Закрыто, ядро выжило** |
| 2 | Нет tail-stratified baseline | 26% top-y + random background | Stratified **24.09** vs SHD **19.20**; 15/15, p = 6.1e-5 | **Закрыто, ядро выжило** |
| 3 | Один реальный датасет | Beijing PM2.5, time split, 20 seeds | Critical 231.0 vs KT 245.7 (13/20, p = 0.216); global KRR/GBM значимо хуже (wins 0–4/20) | **Закрыто, отрицательно** |
| 4 | Фиксированный split | 3 splits (42/43/44) × 5 seeds | SHD 19.14 / 20.10 / 18.92; KT 22.99 / 24.20 / 23.29; 15/15 | **Закрыто, ядро выжило** |
| 5 | Один heal | 3-shift A→B→C→D, 5 seeds (scaler fix) | Новые домены: D 80.6→22.0, C 47.0→16.3, B 18.5→9.1; A 3.89→12.49; BW 0.507 | **Закрыто, частично: новые домены учит, forgetting есть** |
| 6 | CPD — внешняя зависимость | `vendor/jarvis_v2_cpd.py` + `jarvis_v2_discovery.py` | Импорт работает, `_fit_gp` доступен | **Закрыто в коде; paper всё ещё без описания CPD** |
| 7 | Нет ablation триггера/RFF | `use_residual` и dual/single-scale, 10 seeds | Residual off лучше critical (SC −0.80; reactor −0.14); dual-scale n.s. | **Закрыто, заявки падают** |
| 8 | Нет ROC/AUC радара | Graded bands + A/B, 5 seeds, B=64/256/1024 | AUC = 1.0, TPR@5% = 1.0, FPR@tau = 0 | **Закрыто, но задача лёгкая** |
| 9 | Нет multiple-comparison correction | Holm по 24 тестам на датасет | Critical p_holm = 4.6e-5; KRR-all vs SKT 0.051 | **Закрыто, один claim пограничный** |
| 10 | Ablation 5 seeds без CI | 10 seeds + bootstrap 95% CI | Kappa=0 +0.668 [0.397, 0.957]; high_band=0.1 +0.752 [0.521, 1.285] | **Закрыто, «single largest» неверно** |

## 1. KT subsample confound

`bench_gaps.py` строит SHD на **том же** i.i.d. subsample из 4 800 точек,
который получают KT/SKT. 3 split × 5 seeds, M = 300:

| Arm | KRR Crit (med / IQR) | KRR All | GBM Crit | Rare captured |
| :--- | :---: | :---: | :---: | :---: |
| SHD full | 19.204 / 0.782 | 15.879 | 18.514 | 64.0 |
| **SHD_sub4800** | **18.714 / 1.037** | 15.992 | **17.519** | **66.0** |
| KT | 23.194 / 1.716 | 16.874 | 23.182 | 58.0 |
| SKT | 21.698 / 2.299 | 16.287 | 21.834 | 55.0 |
| Stratified | 24.092 / 2.431 | 16.612 | 22.885 | 53.0 |
| Random | 23.959 / 3.361 | 17.074 | 22.813 | 52.0 |
| K-Means | 23.167 / 2.507 | 17.568 | 23.335 | 32.0 |

Paired tests (15 records):

- SHD vs KT: **15/15**, p = 6.1e-5, Holm = 0.00256.
- SHD vs SKT: **15/15**, p = 6.1e-5, Holm = 0.00256.
- **SHD_sub4800 vs KT: 15/15, p = 6.1e-5.**
- SHD full vs SHD_sub4800: 7/15, p = 0.389 — разницы нет.

Вывод: победа над KT/SKT **не** объясняется тем, что SHD видит 17 010 точек.
SHD с 4 800 точек даже чуть лучше full-SHD; метод data-efficient.

## 2. Tail-stratified baseline

Простой baseline: 26% бюджета из top-26% y + 74% random background
(с Horvitz–Thompson весами он вырождается в тот же uniform, потому что
стратифицированная выборка самовзвешенная).

- SHD 19.204 vs stratified 24.092; **15/15**, p = 6.1e-5.
- На split 43 stratified ближе (22.48 vs SHD 20.10), но всё равно хуже.

Вывод: tail-stratification не закрывает разрыв; фазовое многообразие и
Voronoi-геометрия дают не только «взять больше высоких y».

## 3. Второй реальный датасет: Beijing PM2.5

43 824 сырых строк → **41 757** полных после удаления NA. Time-ordered split
80/20 (тест — будущий 2014 год). Critical = top 3% train PM2.5 (threshold
**324 µg/m³**; **374** critical test rows). 20 seeds:

| Baseline | KRR Crit | SHD wins | p (raw / Holm) | KRR All | SHD wins | p |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| Random | 248.92 | 15/20 | 0.0121 / 0.0966 | 88.24 | **3/20** | 3.9e-4 (SHD worse) |
| K-Means | 241.18 | 11/20 | 0.277 / — | 88.92 | **4/20** | 4.8e-4 (SHD worse) |
| KT | 245.74 | 13/20 | 0.216 / — | 89.89 | **2/20** | 8.2e-5 (SHD worse) |
| SKT | 246.02 | 14/20 | 0.0328 / 0.229 | 86.53 | **1/20** | 1.0e-5 (SHD worse) |
| **SHD** | **231.01** | — | — | **95.08** | — | — |

**Исправление после перепроверки:** исходный прогон фильтровал `cbwd == "CV"`,
но в датасете категория записана как **`cv`**; это выбрасывало 9 387 строк
(21%) и искажало результат. После фикса n = 41 757, критический порог 324,
и картина меняется: SHD **незначимо** лучше на critical (231.0 vs 245.7,
13/20, p = 0.216) и **значимо хуже на global** (KRR-all 95.1 vs 86.5–89.9;
GBM-all 86.1 vs 78.3–80.0; wins 0–4/20). SHD захватывает 10.0 rare-точек
из 300 при ожидании ~9 (3% critical) — то есть никакого rare-over-representation
на PM2.5 нет: дефолтная полоса 26% слишком широка для 3% режима.

Вывод: **PM2.5 — отрицательный результат**, а не neutral. Метод не
переносится на этот time-shifted sensor-стрим в дефолтной конфигурации; нужен
более узкий trigger (например, 97-й перцентиль) или адаптивная ширина полосы,
и это отдельный эксперимент. «State-of-the-art на experimental physics
streams» — слишком широко.

## 4. Множественные splits

3 split seeds × 5 coreset seeds. SHD crit по split'ам: 19.139 / 20.095 / 18.919 K;
KT: 22.991 / 24.203 / 23.294 K; SKT: 21.099 / 23.737 / 21.446 K.
Во всех трёх split'ах SHD выигрывает; pooled 15/15. Фиксированный split не
объясняет результат.

## 5. Lifelong: 3-shift heal

A (<20 K) → B (20–40) → C (40–77) → D (≥77), 5 seeds, M base 200,
inject 45, cap 260. **Перепроверка:** первый прогон этого бенчмарка не
стандартизировал признаки (`fit_eval` тренировал KRR/GBM на raw X), что
искусственно убивало новые домены. После фикса (StandardScaler на crystal,
как в heal/bench_seeds) картина другая:

| Stage | A KRR | B KRR | C KRR | D KRR | BW after | M |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| A only | 3.89 | 18.48 | 47.01 | 80.58 | — | 200 |
| heal B | 5.90 | **9.08** | 35.40 | 66.92 | 0.439 | 245 |
| heal C | 9.74 | 14.03 | **16.29** | 41.30 | 0.526 | 260 |
| heal D | 12.49 | 20.89 | 17.13 | **22.00** | 0.507 | 260 |

Вывод: heal **учит новые домены** (D 80.6 → 22.0, C 47.0 → 16.3,
B 18.5 → 9.1), но **catastrophic forgetting накапливается**: A 3.89 → 12.49
(×3.2), B после C/D 9.08 → 20.89, C после D 16.29 → 17.13. Паспорт не
сходится (BW ~0.5 > 0.085). То есть «lifelong» **частично подтверждён**
(новые домены выучиваются), но «zero catastrophic forgetting» — **нет**.
Предыдущий вывод «lifelong не работает» был артефактом отсутствия scaler;
правильный вывод: heal работает, но требует retention-механизма.

## 6. CPD dependency

`vendor/jarvis_v2_cpd.py` и `vendor/jarvis_v2_discovery.py` скопированы
из JARVIS Lab; `bridge_neon_bo.py` сначала ищет их в `vendor`, затем в
соседнем репозитории. Проверено: модуль импортируется, `_fit_gp is not None`,
`discover_phase_boundary_cpd` доступен. CPD-числа теперь воспроизводимы из
одного SHD-репо. В paper всё ещё нет описания CPD-Rank — это отдельный
текстовый долг.

## 7. Trigger и RFF ablation

### Residual trigger (`use_residual`)

| Dataset | Residual ON (crit) | OFF (crit) | Diff (95% CI) | All ON → OFF |
| :--- | :---: | :---: | :---: | :---: |
| UCI SC, 5 seeds | 19.139 | **17.961** | **−0.802 [−1.300, −0.416]** | 15.366 → 15.777 (+0.386 [0.035, 0.475]) |
| Reactor, 10 seeds | 3.440 | **3.316** | **−0.141 [−0.213, −0.030]** | 0.976 → 0.981 (n.s.) |

Residual **ухудшает** critical RMSE и **улучшает** global RMSE на SC. Это
global-vs-critical trade-off. Paper'у нельзя говорить, что residual
«directs phase-peak reservation»; правильнее: y-only trigger лучше для
rare-regime, residual — для global fit.

### Dual-scale RFF (reactor, 10 seeds)

| Config | KRR Crit | Diff vs dual (95% CI) |
| :--- | :---: | :---: |
| Dual (0.75 / 2.10) | 3.440 | — |
| Single low (0.75 / 0.75) | **3.211** | −0.192 [−0.399, +0.034] |
| Single high (2.10 / 2.10) | 3.279 | −0.040 [−0.328, +0.078] |

Разница не значима; dual-scale не показывает измеримой пользы. Заявление
«dual-scale RFF» надо либо убрать, либо назвать design choice без выигрыша.

## 8. ROC/AUC радара

- `train_cut=40`: Domain A (<40) vs B (≥40), AUC = **1.0**, TPR@5% FPR = 1.0,
  FPR@published τ = 0, B = 64/256/1024, 5 seeds.
- `train_cut=20`: кристалл на <20 K; graded OOD 20–30, 30–40, 40–77, 77+;
  AUC = **1.0** во всех полосах, всех B, 5 seeds.

Радар идеально разделяет; но сдвиг сильный, и задача лёгкая. Для paper:
сообщить AUC = 1.0 с оговоркой; для настоящей характеристики нужен
gradual/noisy shift. Латентность: 4.2 ms (B=64), 21.6 ms (B=1024),
84 ms (B=5780) на этой машине; не «sub-millisecond».

## 9. Multiple comparisons (Holm)

| Family | Tests | Main critical | Borderline |
| :--- | :---: | :---: | :--- |
| SC bench_seeds | 24 | p_holm = 4.58e-5 (vs Random/KMeans/KT/SKT) | KRR-all vs SKT p_holm = 0.051; GBM-all vs KT 0.197 |
| Reactor bench_seeds | 24 | все p_holm ≤ 3.2e-2 | — |
| BO bridge | 16 | все p_holm ≤ 1.5e-2 | — |
| Heal | 24 | все p_holm ≤ 4.6e-5 | — |

Главный critical-claim выживает; KRR-all vs SKT после Holm — пограничный,
его нельзя называть значимым. Paper должен report Holm, а не только raw p.

## 10. Ablation: 10 seeds + bootstrap CI

| Constant | Value | Median | 95% CI | Diff vs default | 95% CI diff |
| :--- | :---: | :---: | :---: | :---: | :---: |
| `phase_kappa` | 0.0 | 4.132 | [3.810, 4.296] | **+0.668** | [0.397, 0.957] |
| | 1.6 | 3.261 | [3.176, 3.396] | −0.119 | [−0.352, +0.051] |
| | 3.2 (default) | 3.440 | [3.250, 3.538] | 0 | — |
| | 6.4 | 3.648 | [3.305, 3.868] | +0.257 | [−0.068, +0.396] |
| `high_band_ratio` | 0.10 | 4.249 | [4.082, 4.478] | **+0.752** | [0.521, 1.285] |
| | 0.26 (default) | 3.440 | [3.250, 3.538] | 0 | — |
| | 0.50 | 2.737 | [2.628, 2.859] | **−0.739** | [−0.800, −0.522] |
| `tail_quantile` | 74 | 3.645 | [3.608, 3.744] | +0.249 | [0.090, 0.538] |
| | 84 (default) | 3.440 | [3.250, 3.538] | 0 | — |
| | 94 | 2.735 | [2.642, 2.789] | **−0.702** | [−0.834, −0.563] |

`high_band_ratio=0.10` хуже, чем `phase_kappa=0`; «single largest degradation»
неверно. `high_band_ratio=0.5` и `tail_quantile=94` значимо лучше дефолта
на reactor — дефолты не оптимальны.

## Что выжило

- UCI SC critical regime: 18.81 vs 22.43 / 22.93 / 23.26 / 21.20; 20/20;
  Holm 4.6e-5.
- **Same-subsample SHD vs KT: 18.71 vs 23.19, 15/15** — главный контроль.
- Tail-stratified baseline: SHD 19.20 vs 24.09, 15/15.
- 3 split × 5 seeds: SHD выигрывает все 15/15.
- Reactor critical: 3.436 vs 4.516 / 4.060; Holm 1.2e-4 / 4.6e-5.
- BO: peak RMSE, EI hits, regret — все выживают Holm.
- Radar: AUC = 1.0; сильная чувствительность.
- One-shift heal: combined 19.10 vs pre 33.86 / naive 21.58; old A degrades,
  но не катастрофически.
- 3-shift heal: новые домены реально выучиваются (D 80.6→22.0, C 47.0→16.3,
  B 18.5→9.1), хотя старый A деградирует 3.89→12.49.

### Контекст: насколько это «rare»

- **SC:** critical (Tc ≥ 77 K) — **18.6%** датасета, не <3%. При M = 300
  ожидание ~56 critical-точек; SHD даёт 64.5 (×1.15). Это **critical-regime
  accuracy**, а не драматический rare-recall.
- **Reactor:** critical — 3%; ожидание ~9; SHD даёт 19.5 (×2.2). Вот здесь
  rare-эффект сильный.
- **PM2.5:** critical — 3%; ожидание ~9; SHD даёт 10.0 (×1.1), то есть
  редкий режим не пере-представлен; дефолтная 26% полоса слишком широка.

Вывод для paper: «rare regime» надо формулировать per-dataset, а не как
универсальное свойство; на SC выигрыш — точность в критическом режиме, на
reactor — реальный rare-recall, на PM2.5 — ни то, ни другое.

## Что не выжило

- **Zero catastrophic forgetting**: 3-shift учит новые домены (D 80.6→22.0),
  но A деградирует 3.89→12.49, B 9.08→20.89; forgetting накапливается.
- **Residual trigger как rare-regime механизм**: выключение улучшает critical.
- **Dual-scale RFF**: нет измеримой пользы.
- **FISTA herding**: существующий one-at-a-time ablation (5 seeds) показывает,
  что `herding_mix=1.0` (без herding) не хуже дефолта (3.326 vs 3.346); заявка
  «kernel herding» как несущая — слабая. Это не входило в 10 пунктов, но
  рецензент это заметит.
- **«phase_kappa — single largest degradation»**: high_band_ratio=0.10 хуже.
- **«guarantees rare cells»**: ablation поддерживает phase geometry, но
  «guarantee» — слишком сильно; правильно «encourages / biases».
- **PM2.5 generalization**: SHD значимо хуже на global KRR/GBM (wins 0–4/20)
  и незначимо лучше на critical (13/20, p = 0.216); дефолтная 26% полоса
  не подходит для 3% режима. Это отрицательный второй датасет, не neutral.
- **8.7% / M=90 / 98 KB / sub-ms / zero catastrophic forgetting** — фактические
  ошибки paper; исправить.
- **KRR-all vs SKT после Holm**: p_holm = 0.051, пограничный; не называть
  значимым.

## Что менять в paper (P0)

1. Abstract: 23.26 / 21.20 / 18.6% / 184 KB / 4–84 ms; убрать sub-ms,
   zero forgetting, «<3%» для SC; заменить на one-shift heal.
2. Table 1/2: Oracle 9 000 для обоих датасетов; Wins column semantics;
   rare capture 59.0 / 55.0.
3. Добавить data-access control: SHD full vs SHD_sub4800 vs KT/SKT.
4. Добавить tail-stratified baseline и 3-split robustness.
5. Добавить PM2.5 (20 seeds) с **отрицательным** результатом; не neutral.
6. Heal: pre-heal 7.32 / 7.10; naive B 23.83 / 16.24; post-heal BW 0.465;
   M = 220; latency 50 ms; убрать «strictly preserved»/«total mitigation».
7. Lifelong: добавить corrected 3-shift (новые домены учит, но forgetting
   накапливается); убрать «zero catastrophic forgetting»; описать retention.
8. Ablation: 10-seed CI; kappa не largest; residual/dual-scale — trade-off.
9. CPD: добавить описание CPD-Rank и ссылку на vendored код.
10. Limitations: KT subsample (закрыт, но disclosed), fixed split vs 3 splits,
    synthetic reactor, PM2.5, Holm, OOD threshold calibration, CPD.

## Артефакты

- `bench_gaps.py` → `bench_gaps_results.json` (3 splits × 5 seeds).
- `bench_pm25.py` → `bench_pm25_results.json` (20 seeds).
- `bench_lifelong.py` → `bench_lifelong_results.json` (5 seeds).
- `bench_radar_roc.py` → `bench_radar_roc_results.json` (cut 20/40).
- `bench_rff.py` → `bench_rff_results.json` (10 seeds).
- `stats_holm.py` → `stats_holm_results.json`.
- `stats_ablation_ci.py` → `bench_ablation_ci_summary.json`,
  `bench_residual_ablation_ci.json`.
- `bench_ablation.py` (10 seeds, per-seed records) → `bench_ablation_ci.json`,
  `bench_residual_ablation.json`, `bench_residual_ablation_sc.json`.
- `vendor/` — CPD/GP code, unmodified.
- Тесты: 21/21 passed после правок v3.9.1.

## Ограничения аудита

- 3 splits × 5 seeds — сигнал, не финальные 20×3; для paper лучше 20 seeds
  на split, но выводы уже устойчивы (15/15).
- PM2.5: 20 seeds, time split; после перепроверки исправлен `cbwd` (`cv`,
  не `CV`); дефолтная 26% полоса не соответствует 3% critical; один датасет.
- Lifelong: 5 seeds, один порядок доменов; negative result устойчив по seeds,
  но не проверены другие `n_inject`/cap.
- ROC: AUC насыщен 1.0; нужен более мягкий/noisy shift для кривой.
- CPD: код вендорен, но paper всё ещё без метода; числа не перепрогонялись
  после вендоринга (импорт проверен).
