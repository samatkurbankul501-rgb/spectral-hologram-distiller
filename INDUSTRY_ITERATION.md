# SHD: итерация «industry-best» (2026-10-17)

Цель: сделать SHD лучшим не «вообще», а в конкретной категории —
**мониторинг редких/критических режимов с известным alarm-порогом**, где
пропуск события дороже средней ошибки. Первая итерация дала рабочий
кандидат: **SHD-C (critical-aware) + опциональный global guard / KT-портфель**.

## Итерация v3.14.0: портфель OOF-стекинга + alarm-zone конформаль (2026-10-25)

Закрыты две слабости карты: (1) «глобально уступает SKT/KT» и (4) «нет
неопределённости».

## Итерация v3.14.1: логит на Scania, пресеты, бюджет солвера гейта (2026-10-25)

Закрыты оставшиеся два хвоста карты слабостей: **(2) Scania logit** и
**(3) дефолты дистиллера** + третий — ConvergenceWarning в гейт-логистике.

### (2) Диагноз logit-провала: провал в регуляризации, не в кристалле

Зонд на тех же кристаллах SHD (5 сидов; оракул-логит воспроизводит
README cost 770 бит-в-бит) показал: историческая рука logit C=1.0 (дефолт
sklearn) — aucV 0.831; логит с тюнингом C по валидации — aucV 0.974 (оракул 0.962); гипотеза «сгладить класс-веса» **опровергнута** (aucV 0.799 —
худшая рука). Лучший C=**0.01** во всех сидах — на 2 порядка сильнее
дефолта, то есть исторический провал — это дефолтная регуляризация, а не
кристалл. Полноценные 10-сидные прогоны (`bench_scania_cost.py --tune-logit-c`,
грид C 0.01–10.0 по валидации):
- **Сценарий 10:1:** SHD+logit(tuned) **840.5** — лучшая рука; SHD+HGB 993.0;
  против Random 974.5 / K-Means 914.0. Честно: значимость против baselines на
  10 сидах НЕ достигнута (8/10, p=0.0625; 7/10, p=0.16) — медианы рангуют SHD
  первым, но 10 сидов этого не доказывают; значимы только победы над
  HGB-baselines (8/10, p=0.0195). Тюнинговая рука против исторической
  SHD-руки: 1419.0 → 840.5 (−40.8%), recall 0.679 → 0.868.
- **Официальный 500:10:** tuned logit HGB не догоняет — фиксируется
  (см. §8 README).
- Механизм позитивных строк кристалла не изменился: медиана 25.0 (SHD) против
  7.0 (Random) / 7.5 (K-Means).

### (3) Дефолты дистиллера: пресеты как данные

`shd/presets.py` (реэкспорт из `shd`): таблица §6 README стала
машиночитаемой — 5 пресетов (`default`, `global_first`, `balanced`,
`critical_first`, `sparse_stream_top3pct`) с measured-числами в `source`.
`apply_preset(name, **overrides)` не трогает бюджетные/репроизводимые настройки (`crystal_size`, `sketch_dim`, `random_state`); `validate_presets()` —
дрейф-гард: падает, если пресет уплывает → «уходит» от сигнатуры конструктора
(тесты + verify_claims). Дефолты конструктора **не тронуты**: исторические
артефакты остаются воспроизводимыми бит-в-бит.

### (доп.) ConvergenceWarning в гейте: поднять бюджет или задокументировать — сделано

Зонд сходимости (сид 0): реактор 10/2000, PM2.5 443/2000, суперпроводник
**1228/2000** (у части сидов доходит до лимита — Warning реален, не артефакт
вывода).
Решение в `shd/portfolio.py`: `max_iter=5000`; солвер **lbfgs** — primary
(закоммиченные артефакты v3.14.0 получены lbfgs; сошедшиеся фиты бит-в-бит
идентичны при любом max_iter выше достигнутого — воспроизводимость, не
предпочтение); **liblinear** — детерминированный fallback только при
исчерпании бюджета; статус записывается в атрибут `gate_solver_` (не
ConvergenceWarning и не чтение stderr).
Регрессии: `tests/test_presets_and_gate.py` — 15 тестов (пресеты, дрейф-гард,
детерминизм гейта). Фоновые прогоны: `bench_scania_cost_tunedC_results.json`
(10:1, 10 сидов) и `bench_scania_cost_tunedC_500x10_results.json` (500:10).

## TL;DR

- **CoresetPortfolio (v3.14.0, 20 seeds × 3 датасета, Wilcoxon) — пересчитано
  2026-11-13 с нуля после фикса critical-aware эксперта (critical_threshold +
  auto_band_ratio=True); числа 2026-11-07 сняты.** SC: crit **18.22** vs 18.39
  (SHD-C) — НЕ значимо (p=0.349, ничья), 23.26 (KT) / 21.19 (SKT) / 19.22
  (hard-gate) — их бьёт (p<1e-4); глобаль **15.38** vs 15.46 (p=0.027).
  PM2.5: глобаль **86.63** vs 101.15 (SHD-C, p=1.9e-6) и 89.89 (KT, p=6.4e-3),
  но SKT (86.53) не обойдён (p=0.475, н.с.); crit **224.67** vs 216.57 —
  ничья (p=0.648), SKT (246.02) обойдён (p=9.5e-6). Реактор: портфель ХУЖЕ
  SHD-C по crit (3.32 vs 3.04, p=3.8e-6), глобаль — ничья (0.975 vs 0.969,
  p=0.70), остальных (KT/SKT/hard-gate) всё равно бьёт (p<1e-4). Итог: после
  фикса портфель — НЕ Pareto-лучший арм (глобальные победы на SC и PM2.5,
  ничьи по crit на SC/PM2.5, проигрыш по crit на реакторе). Soft OOF-гейт читает ОБА предсказания и
  бьёт пороговый hard-gate на всех трёх датасетах — схема v3.13 (gate на
  прокси) была принципиально слабее. (GBM-колонка портфеля дублирует арм
  SHD-C по построению — оба фитят один и тот же критический coreset с одним
  seed'ом — поэтому таблица Wilcoxon только по KRR.)
- **ConformalCrystal (v3.14.0):** split-conformal интервалы вокруг любой
  модели; калибровать на СРЕЗЕ ПОТОКА, не на строках кристалла
  (y-стратификация ломает обменность: crit-покрытие 0.23 при цели 0.90 —
  измерено). `mondrian="alarm_zone"`: группа `pred >= threshold − delta`,
  delta автоподбором под `zone_fraction` (0.03 дефолт; 0.10 ловит 116/122
  истинных пиков реактора при in-zone покрытии 0.946; 0.20 — дрейф 0.918:
  шире ≠ лучше). Двусторонние квантили SIGNED-остатка в группе дают точное
  1−α внутри зоны даже при систематическом занижении пиков моделью
  (регрессия к среднему). Своп |resid|/signed (маргинал 0.859) и ранг
  1−α+α/2 (0.859/0.906) — исправленные дефекты первых версий.
- **Портфель на реакторе — не дефолт:** после фикса critical-aware эксперта портфель на реакторе
  ХУЖЕ одиночного SHD-C по crit (3.32 vs 3.04, p=3.8e-6; глобаль — ничья, 0.975 vs 0.969,
  p=0.70) — он не «не мешает», а стоит качества на crit; рекомендация opt-in,
  дефолт — одиночный SHD-C.

## TL;DR (итерация 2026-10-17)

- **SHD-C (critical-aware band)** — новое режим: если известен порог
  `critical_threshold`, фазовая полоса строится по `y >= threshold`, а бюджет
  ячеек — `clip(2·critical_fraction, 10%, 30%)`.
- **SC (20 seeds):** SHD-C **18.390** crit vs base 18.805; rare **72.5** vs 64.5;
  KRR-all 15.463 vs 15.423 (+0.04), GBM-all 16.087 vs 16.065 (+0.02).
  То есть SHD-C — critical/rare win с минимальным global-компромиссом,
  а не полное доминирование (5-seed 18.20/15.34 был оптимистичен).
- **PM2.5 (20 seeds):** SHD-C **216.6** crit vs base 231.0; rare **25** vs 10;
  но global 101.2 vs 95.1 (+6.1), GBM-all 91.3 vs 86.1. Critical/rare win
  с ценой global; 5-seed 207.98 был оптимистичен.
- **Global guard** (50–100 K-Means medoids) возвращает global:
  при M=350 critical 206.6 / all 92.2 / GBM 85.0;
  при M=400 critical 211.5 / all 90.0 / GBM 83.8.
- **Портфель SHD-C + KT, gate q90 (20 seeds):** critical **200.5**,
  all 91.0, GBM **79.2** — лучше SHD (216.6/101.2/91.3) по всем трём,
  лучше KT (245.0/87.1/78.9) по critical, global на 3.9 K хуже KT.
- **SHD-C + 300 KMeans guard, M=600 (5 seeds):** critical **196.0**,
  all **85.3**, GBM 78.9, rare 29 — лучший rare-regime баланс;
  против KMeans M=600 (214.7/83.1/72.7): critical −18.7, global +2.2,
  GBM +6.2. Это главный 600-точечный кандидат (нужны 20 seeds).
- **Lifelong:** после исправления scaler heal реально учит новые домены
  (D 80.6→22.0, C 47.0→16.3, B 18.5→9.1), но старый A деградирует
  3.89→12.49; boost/protect лишь двигают trade-off. «Zero forgetting» — нет.
- **SHD-Bank:** отдельный кристалл на домен + radar-router: routing
  **100%** (band и per-window 256), A 3.9 / B 5.1 / C 11.3 / D 11.3
  vs single-A 3.9 / 18.5 / 47.3 / 81.2; 800 точек; forgetting нет.
- **Универсального SOTA нет:** на чистом global RMSE KT/SKT всё ещё лучше;
  на PM2.5 single-crystal SHD проигрывает global. Заявлять «best in industry»
  можно только для rare-regime ниши и с портфелем/guard.

## 1. SHD-C: critical-aware phase band

Новый API:

```python
SpectralHologramDistiller(
    ...,
    critical_threshold=77.0,   # known alarm threshold, e.g. Tc >= 77 K
    auto_band_ratio=True,      # cells = clip(2*critical_fraction, 10%, 30%)
)
```

Механика: при сборке lattice `is_high = y >= critical_threshold` вместо
`score >= percentile(score, tail_q)`; `m_high = round(M * clip(2*frac, 0.10, 0.30))`.
Дефолт (`critical_threshold=None`) сохраняет старое поведение бит-в-бит.

### SC (5 seeds, M=300)

| Config | KRR Crit | KRR All | GBM All | Rare |
| :--- | :---: | :---: | :---: | :---: |
| base (t84/h26) | 19.14 | 15.37 | 16.47 | 64 |
| **SHD-C auto** | **18.20** | **15.34** | **16.09** | **73** |
| SHD-C h26 | 18.67 | 15.57 | 15.96 | 63 |
| SHD-C h10 | 21.87 | 15.70 | 16.49 | 31 |

SHD-C auto — лучший по critical/rare; на 20 seeds KRR-all/GBM почти
не меняются (см. ниже). 10% слишком узко для SC, где critical = 18.6%.

**20-seed confirmation SC:** crit **18.390** (IQR 0.660), KRR-all
15.463 (IQR 0.233), GBM-all 16.087 (IQR 0.629), rare 72.5 (IQR 4.0).
Base 20 seeds: 18.805 / 15.423 / 16.065 / 64.5. Итог: critical −0.415,
rare +8, global +0.04 — честный trade-off, не «доминирование».

### SC, budget-matched M=350 (5 seeds)

| Arm | KRR Crit | KRR All | GBM All | Rare |
| :--- | :---: | :---: | :---: | :---: |
| Random | 23.20 | 16.65 | 16.32 | 63 |
| K-Means | 23.36 | 17.46 | 17.52 | 34 |
| KT | 22.91 | 16.72 | 16.46 | 67 |
| SKT | 21.27 | 15.57 | **15.66** | 65 |
| **SHD** | **18.84** | **15.33** | 16.06 | **76** |
| SHD + guard 50 | 18.60 | 15.49 | 15.95 | 68 |

При равном бюджете SHD лучший по critical и KRR-all; GBM отстаёт от SKT
на 0.29–0.40 K. SHD-C (critical-aware, M=300) даёт 18.20/15.34/16.09 —
то есть даже с меньшим бюджетом лучше M=350 baselines по critical/all.

### PM2.5 (5 seeds, M=300)

| Config | KRR Crit | KRR All | GBM All | Rare |
| :--- | :---: | :---: | :---: | :---: |
| base (t84/h26) | 235.64 | 94.87 | 88.89 | 10 |
| **SHD-C auto/h10** | **207.98** | 98.75 | 87.85 | **25** |
| SHD-C h26 | 207.09 | 99.94 | 102.12 | 39 |
| SHD-C h05 | 222.76 | 104.85 | 88.12 | 14 |

Critical −27.7 K и rare ×2.5; global +3.9 K. `h26` ломает GBM (102.1) —
слишком много high-y точек; auto min 10% — правильный компромисс.

**20-seed confirmation PM2.5:** SHD-C **216.6 / 101.2 / 91.3 / 25** vs base
231.0 / 95.1 / 86.1 / 10. Critical −14.4 K, rare +15, global +6.1 K —
5-seed 207.98 был оптимистичен; критический выигрыш сохраняется, цена
global выше. Поэтому для PM2.5 нужен guard/портфель.

## 2. Global guard: 50–100 K-Means medoids

SHD-C crystal M=300 + K-Means global-guard medoids (append, weights normalized).

### PM2.5, budget-matched (5 seeds)

| M | Arm | KRR Crit | KRR All | GBM All | Rare |
| :---: | :--- | :---: | :---: | :---: | :---: |
| 350 | K-Means | 234.52 | **87.51** | **78.94** | 7 |
| 350 | KT | 247.90 | 84.41 | 75.68 | 10 |
| 350 | SKT | 253.39 | 85.02 | 73.47 | 9 |
| 350 | SHD-C + guard 50 | **206.6** | 92.2 | 85.0 | 25 |
| 400 | K-Means | 255.25 | 89.21 | 81.44 | 9 |
| 400 | KT | 259.43 | 85.79 | 78.78 | 10 |
| 400 | SKT | 242.52 | 83.75 | 76.82 | 11 |
| 400 | SHD-C + guard 100 | **211.5** | 90.0 | 83.8 | 27 |

Вывод: guard 100 при M=400 почти закрывает global-разрыв с K-Means
(90.0 vs 89.21) и GBM (83.8 vs 81.44), сохраняя critical 211.5 vs 255.25.
Против SKT global/GBM ещё хуже (83.75/76.82), но critical лучше на 31 K.

### PM2.5, equal budget M=600 (5 seeds)

| Arm | KRR Crit | KRR All | GBM All | Rare |
| :--- | :---: | :---: | :---: | :---: |
| Random | 232.40 | 86.26 | 75.17 | 19 |
| K-Means | 214.71 | **83.14** | 72.74 | 14 |
| KT | 245.38 | 86.67 | 74.50 | 19 |
| SKT | 229.87 | 85.17 | **71.67** | 17 |
| SHD M=600 | **180.46** | 93.29 | 78.91 | 39 |
| SHD + 300 guard | 198.67 | 87.64 | 75.51 | 15 |
| **SHD-C + 300 guard** | **196.0** | 85.3 | 78.9 | 29 |

При равном M=600: SHD-C+guard — лучший critical (196.0; −18.7 к K-Means,
−49.4 к KT), global 85.3 (+2.2 к K-Means), GBM 78.9 (+6.2 к K-Means,
+7.2 к SKT). Это лучший rare-regime баланс; GBM остаётся слабым местом.
Это **rare-regime специалист**, не универсальный SOTA.

## 3. Портфель SHD-C + KT (gate q90)

Два эксперта: SHD-C (critical) + KT (global). Gate: если SHD-предсказание
`>= q90(train y)`, использовать SHD, иначе KT. **20 seeds**, PM2.5:

| Arm | KRR Crit | KRR All | GBM All |
| :--- | :---: | :---: | :---: |
| SHD-C | 216.6 | 101.2 | 91.3 |
| KT | 245.0 | **87.1** | **78.9** |
| SHD-C + KT, gate q90 | **200.5** | 91.0 | 79.2 |
| SHD-C + KT, gate crit | 227.3 | 87.6 | 79.2 |

Портфель q90 **доминирует SHD** по всем трём и **KT** по critical;
global на 3.9 K хуже KT, GBM на 0.3 K хуже. 5-seed результат (199.1)
подтвердился на 20 seeds (200.5), в отличие от gate по crit (227.3) —
важно, что gate q90, а не жёсткий порог. Артефакт — два кристалла
(600 точек, ~350 KB), CPU. На SC портфель **не помогает**: SHD-C один
лучше (critical 18.2 vs 22.2 у gate q90, 5 seeds); там fallback не нужен.

## 4. Lifelong: что реально происходит

Первый lifelong-бенчмарк не стандартизировал признаки — это был баг
бенчмарка, а не метода. После `StandardScaler` (5 seeds, M base 200,
inject 45, cap 260):

| Stage | A | B | C | D | BW | M |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| A only | 3.89 | 18.48 | 47.01 | 80.58 | — | 200 |
| heal B | 5.90 | 9.08 | 35.40 | 66.92 | 0.439 | 245 |
| heal C | 9.74 | 14.03 | 16.29 | 41.30 | 0.526 | 260 |
| heal D | 12.49 | 20.89 | 17.13 | 22.00 | 0.507 | 260 |

Heal **учит новые домены** (D 80.6→22.0, C 47.0→16.3, B 18.5→9.1), но
старый A деградирует 3.89→12.49 (×3.2), B после C/D 9.08→20.89.
Boost 4 (n=90, cap=320): D 15.66, A 15.45 — новые лучше, старые хуже.
Retention (boost 2, protect_old_weight_frac=0.5): D 17.59, A 14.09 —
посередине. **Zero forgetting нет**; нужен domain bank / reservoir.

### Domain bank (SHD-Bank): zero forgetting by construction

Реализовано в `shd.DomainBank` (`shd/bank.py`): nearest-passport routing поверх
нескольких `.hologram.npz`; банк не перезаписывает старые кристаллы. Плюс
`heal_domain_shift(..., passport_recalibrate=True)` для случая, когда старый домен
больше не вернётся.

Вместо слияния — отдельный кристалл на домен + radar-router. 4 домена
(A/B/C/D), M=200 каждый, 5 seeds:

| Arm | A | B | C | D | Routing |
| :--- | :---: | :---: | :---: | :---: | :---: |
| Single A crystal | 3.9 | 18.5 | 47.3 | 81.2 | — |
| **SHD-Bank (oracle/radar)** | **3.9** | **5.1** | **11.3** | **11.3** | **100%** |

Radar-router выбрал правильный кристалл в **20/20** band-seed случаях;
A не деградирует вообще, D 11.3 vs 81.2 у single-A. Артефакт — 4×200=800
точек (~0.5 MB). Это вероятный ответ на «lifelong»: не heal одного
кристалла, а **bank + router**. Ограничение: routing проверен на
band-level окнах; для смешанного потока нужен per-window router.

**Per-window router (256 rows):** routing accuracy **100%**, RMSE
идентичны oracle (A 3.9 / B 5.1 / C 11.3 / D 11.3). То есть bank
работает и на оконном уровне, не только на целом band.

## 5. Что не сработало

- **Узкие tail-полосы без threshold** на PM2.5: `t97_h05` crit 260.7,
  all 101.4 — хуже base. Score-percentile не равен истинному critical.
- **Hard gate на KMeans** (без KT): MoE теряет critical (243.9 vs SHD 221.0).
- **Single-crystal SHD** на PM2.5: global 95–102 vs baselines 84–89.
- **Hybrid swap50** на SC: critical/all хуже base; польза только в GBM.
- **`phase_kappa`/residual**: residual выключение улучшает critical, но
  ухудшает global; это knob, не free win.

## 6. Рекомендуемая конфигурация для industry

**Для known-threshold rare-regime monitoring:**

```python
SHD-C: critical_threshold=alarm_threshold, auto_band_ratio=True
       (cells = 10–30% in proportion to critical fraction)
+ 50% global guard: 300 K-Means medoids (M=600 total)
+ optional KT fallback with gate on SHD prediction q90 (600-point portfolio)
```

- SC: SHD-C alone; critical 18.39 vs 18.81, rare 72.5 vs 64.5,
  global +0.04 (почти без изменений).
- PM2.5: **SHD-C + 300 guard** (critical 196.0, global 85.3, GBM 78.9;
  5 seeds) — лучший rare-regime баланс; портфель q90 (200.5/91.0/79.2;
  20 seeds) — альтернатива с более надёжным 20-seed подтверждением.
- Artifact: 300–600 medoids, ~180–350 KB; CPU; radar unchanged.
- **Не заявлять universal SOTA**: на чистом global RMSE KT/SKT лучше.
  Заявлять **best-in-class для rare-regime** с асимметричной ценой ошибки.

## 7. Что делать дальше (приоритет)

1. **20 seeds SHD-C:** сделано (SC 18.390; PM2.5 216.6).
   **20 seeds SHD-C + 300 guard** заблокировано UCI throttling (PM2.5 CSV
   таймаутит); сейчас 5 seeds. Нужен локальный кэш PM2.5 и повтор.
2. **Третий реальный датасет:** Appliances ZIP тестово скачался, но UCI
   throttles (12 MB / ~3 KB/s); benchmark не завершён. Повторить с локальным
   кэшем или взять Metro Interstate Traffic / NASA CMAPSS / Gas Sensor Drift.
3. **Domain bank:** сделано (100% routing, A 3.9/B 5.1/C 11.3/D 11.3);
   следующий шаг — per-window router на смешанном потоке.
4. **Soft gate / stacking** вместо hard q90: взвешенное смешение SHD/KT
   в переходной зоне; может убрать 3–4 K global-разрыва.
5. **Продукт:** `SHD-C` API + `critical_threshold`; guard; bank;
   latency/artifact table; one-command benchmark suite.

## Артефакты

- `bench_portfolio.py` → `bench_portfolio_{pm25,superconduct,reactor}_results.json`
  (20 seeds; arms shd/kt/skt/portfolio/hard_gate + Wilcoxon)
- `tests/test_portfolio_conformal.py` — 13 регрессий портфеля и конформаля
- `bench_shdc.py` → `bench_shdc_{superconduct,pm25}_results.json` (20 seeds)
- `bench_bank.py` → `bench_bank_results.json` (domain bank)
- `bench_passport.py` → `bench_passport_results.json` (blend / recalibrate / bank)
- `bench_weights.py` → `bench_weights_*_results.json` (gamma vs uniform)
- `bench_critical_guard.py` → `bench_critical_guard_results.json`,
  `bench_critical_guard100_results.json`,
  `bench_critical_guard300_results.json`
- `bench_pm25_hybrid.py` → `bench_pm25_hybrid_results.json`
- `bench_pm25_sweep.py` → `bench_pm25_sweep_results.json`
- `bench_budget.py` → `bench_budget_{superconduct,pm25}_results.json`,
  `bench_budget_pm25_m400_results.json`,
  `bench_budget_pm25_m600_results.json`
- `bench_moe_kt.py` → `bench_moe_kt_results.json`,
  `bench_moe_kt_superconduct_results.json`
- `bench_appliances.py` — третий реальный датасет (сеть/парсинг, перезапуск)
- `bench_lifelong.py` → `bench_lifelong_results.json`,
  `bench_lifelong_boost_results.json`,
  `bench_lifelong_retention_results.json`
- Новые параметры: `critical_threshold`, `auto_band_ratio` (`shd/distiller.py`),
  `new_weight_boost`, `passport_blend_min/max`, `protect_old_weight_frac`
  (`heal_domain_shift`), `use_residual`.
- Тесты: **21/21 passed**.
