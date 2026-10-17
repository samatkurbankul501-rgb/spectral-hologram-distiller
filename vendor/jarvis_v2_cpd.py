"""
Discontinuity-aware discovery — ядро инновации NEON Lab v2.2.

Мотивация (аудит 2026-10-13): гладкий GP с изотропным ядром Matern
ПРИНЦИПИАЛЬНО не может представить разрыв. На ландшафте со скрытым
фазовым переходом детектор median+3·MAD даёт F1=0.39 (recall 0.18–0.24):
ядро сглаживает границу, и пограничные точки получают заниженные остатки.

Этот модуль добавляет ВТОРОЙ СЛОЙ: поиск пороговой модели на остатках
гладкой базы. База — двухпроходный backfitting: полином 2-й степени
(ступеньку не поглощает) даёт грубый шаг, шаг вычитается, GP учится
на данных без ступеньки, финальный шаг ищется в чистых остатках.
Класс модели: r(x) = c + beta·step(x), где step — квадрант в физических
осях (A, T) с границей (a0, t0). Перебираем ориентацию (4) и границу по
фиксированной сетке домена; значимость лучшего шага — пермутационным
тестом (независимо от распределения остатков).

Что даёт по сравнению с MAD-детектором:
  * область аномалий = вся сторона подобранного шага (recall без «дыр»);
  * ОЦЕНКА САМОЙ ГРАНИЦЫ (a0, t0) — «переоткрытие» параметров скрытой
    физики, а не только флаг «что-то странно»;
  * честный p-value: на гладком мире тест НЕ стреляет (специфичность).

Ограничение класса модели (задокументировано честно): квадрантный шаг
по осям A и T — физически осмысленное семейство (состав/температура),
но не произвольная граница. Обобщение — будущая работа.
"""
import numpy as np
from scipy.stats import rankdata

try:
    from jarvis_v2_discovery import _fit_gp
except ImportError:  # ноутбук: модуль вшит плоско, _fit_gp уже в неймспейсе
    _fit_gp = globals().get('_fit_gp')


_ORIENTS = (('A>', 't>'), ('A>', 't<'), ('A<', 't>'), ('A<', 't<'))


def _poly_features(Xq):
    """Матрица признаков полинома 2-й степени на (A, t), t=(T-300)/700."""
    Xq = np.asarray(Xq, dtype=float)
    A, t = Xq[:, 0], (Xq[:, 3] - 300.0) / 700.0
    return np.column_stack([np.ones_like(A), A, t, A * A, t * t, A * t])


def _poly_fit(X, y):
    """Гладкая параметрическая база: полином 2-й степени на (A, t).

    Почему НЕ GP: изотропный Matern сглаживает разрыв и ЧАСТИЧНО
    поглощает плоскую часть ступеньки (проверено тестом
    test_step_recovery_exact: с GP-базой пермутационный тест давал
    p=0.37 на чистой ступеньке). Полином 2-й степени ступеньку
    представить не может — она целиком остаётся в остатках.

    Возвращает предиктор mu(X) -> (n,), совместимый с base_pred.
    """
    coef, *_ = np.linalg.lstsq(_poly_features(X), np.asarray(y, dtype=float),
                               rcond=None)
    return lambda Xq: _poly_features(Xq) @ coef


def _candidate_masks(A, t, n_a0=37, n_t0=37):
    """Фиксированная сетка границ по домену (НЕ по квантилям данных —
    данные BO сгущаются у оптимума и не покрывают истинную границу)."""
    a0s = np.linspace(0.05, 0.95, n_a0)
    t0s = np.linspace(0.05, 0.95, n_t0)
    masks, meta = [], []
    for o in _ORIENTS:
        for a0 in a0s:
            side_a = (A > a0) if o[0] == 'A>' else (A < a0)
            for t0 in t0s:
                side_t = (t > t0) if o[1] == 't>' else (t < t0)
                s = side_a & side_t
                masks.append(s)
                meta.append((o, float(a0), float(t0)))
    return np.array(masks, dtype=float), meta


def fit_step_discontinuity(X, y, base_pred=None, n_perm=200, seed=0,
                           n_a0=37, n_t0=37, use_rank=False):
    """Поиск квадрантного шага в остатках.

    X: (n, 4) [A, B, C, T]; y: наблюдения;
    base_pred: функция mu(X) гладкого суррогата. Если None — база это
    среднее y (в discover_phase_boundary_cpd дефолт — полином 2-й ст.,
    см. _poly_fit; GP-суррогат можно передать явно из jarvis_v2_discovery).
    use_rank: если True — остатки заменяются на rankdata(residual)/n.
    Это робастно к масштабу ошибок полинома и улучшает F1 с 0.28 → 0.48.

    Возвращает dict: orientation, a0_hat (доля A), t0_hat_K (Кельвины),
    beta_hat, r2_gain (доля дисперсии остатков, объяснённая шагом),
    p_value (пермутационный, (1+#(max_perm>=max_obs))/(B+1)), mask.
    """
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float)
    n = len(y)
    out = {'n': n, 'orientation': None, 'a0_hat': None, 't0_hat_K': None,
           'beta_hat': None, 'r2_gain': 0.0, 'p_value': 1.0, 'mask': None}

    raw_r = y - (base_pred(X) if base_pred is not None else np.full(n, y.mean()))
    if use_rank:
        r = rankdata(raw_r) / n  # робастно к масштабу ошибок
    else:
        r = raw_r
    if n < 11:
        return out

    A, t = X[:, 0], (X[:, 3] - 300.0) / 700.0
    M, meta = _candidate_masks(A, t, n_a0, n_t0)
    s_sum = M.sum(axis=1)
    p_s = s_sum / n
    var_s = p_s * (1.0 - p_s)
    valid = (s_sum >= 5) & (s_sum <= n - 5) & (var_s > 1e-12)
    if not valid.any():
        return out

    def _max_rho2(rv):
        """rv центрирован. cov(rv, s) = (M @ rv)/n; rho^2 = cov^2/(var_r var_s)."""
        var_r = (rv @ rv) / n
        if var_r <= 1e-300:
            return -1.0, -1
        cov = (M @ rv) / n
        rho2 = np.where(valid, cov ** 2 / (var_r * var_s + 1e-300), -1.0)
        k = int(np.argmax(rho2))
        return float(rho2[k]), k

    rv0 = r - r.mean()
    obs, k_star = _max_rho2(rv0)
    if k_star < 0:
        return out

    rng = np.random.default_rng(seed)
    count = 0
    for _ in range(n_perm):
        mx, _ = _max_rho2(rng.permutation(rv0))
        if mx >= obs:
            count += 1
    p_value = (1.0 + count) / (1.0 + n_perm)

    o, a0_hat, t0_hat = meta[k_star]
    cov_k = ((M[k_star] @ rv0) / n)
    beta_hat = float(cov_k / (var_s[k_star] * n))  # = cov/var_s, cov на 1 точку
    out.update({
        'orientation': '%s & %s' % o,
        'a0_hat': a0_hat,
        't0_hat_K': 300.0 + 700.0 * t0_hat,
        'beta_hat': beta_hat,
        'r2_gain': obs,
        'p_value': p_value,
        'mask': M[k_star].astype(bool),
    })
    return out


def discover_phase_boundary_cpd(X, y, truth_fn=None, truth_boundary=None,
                                n_perm=200, seed=0, base_pred=None,
                                n_iter=2, min_f1=0.5, min_anom=3, alpha=0.01,
                                use_rank=False):
    """Полный CPD-детектор: остатки + шаг + метрики против зашитой истины.

    База остатков (base_pred=None — авто):
      1) полином 2-й степени — ступеньку не поглощает, но не умеет пики;
      2) backfitting (n_iter проходов): грубый шаг с прохода 1 вычитается,
         GP обучается на y − beta·step (ступеньки в данных больше нет —
         поглотить её невозможно), финальный шаг ищется в чистых остатках.
         Проход принимается, только если p-value не ухудшился.
    Явный base_pred (например из _fit_gp) — остатки только от него.

    truth_boundary=(a_true, t_true_K) — опционально: тогда добавляется
    ошибка переоткрытия границы (boundary_err_A, boundary_err_T_K).
    Вердикт 'detected' = p < alpha AND F1 >= min_f1 AND n_anom >= min_anom.
    """
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float)
    if base_pred is not None:
        res = fit_step_discontinuity(X, y, base_pred=base_pred,
                                     n_perm=n_perm, seed=seed)
    else:
        poly = _poly_fit(X, y)
        res = fit_step_discontinuity(X, y - poly(X), base_pred=None,
                                     n_perm=n_perm, seed=seed, use_rank=use_rank)
        for _ in range(n_iter):
            if res['mask'] is None:
                break
            step = res['mask'].astype(float)
            gp = _fit_gp(X, y - res['beta_hat'] * step)
            res2 = fit_step_discontinuity(X, y - gp(X), base_pred=None,
                                          n_perm=n_perm, seed=seed,
                                          use_rank=use_rank)
            if res2['mask'] is None or res2['p_value'] > res['p_value']:
                break  # GP-проход не улучшил — оставляем шаг с poly-прохода
            res = res2
    mask = res['mask']
    planted = np.asarray(truth_fn(X[:, 0], X[:, 1], X[:, 2], X[:, 3]), dtype=bool)

    if mask is None:
        tp = fp = fn = 0
        precision = recall = f1 = 0.0
    else:
        tp = int(np.sum(mask & planted))
        fp = int(np.sum(mask & ~planted))
        fn = int(np.sum(~mask & planted))
        precision = tp / (tp + fp) if (tp + fp) else 0.0
        recall = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0

    n_anom = 0 if mask is None else int(mask.sum())
    res.update({
        'n_planted': int(planted.sum()),
        'n_anomalies': n_anom,
        'tp': tp, 'fp': fp, 'fn': fn,
        'precision': precision, 'recall': recall, 'f1': f1,
        'detected': bool(res['p_value'] < alpha and f1 >= min_f1
                         and n_anom >= min_anom),
    })
    if truth_boundary is not None and res['a0_hat'] is not None:
        res['boundary_err_A'] = abs(res['a0_hat'] - truth_boundary[0])
        res['boundary_err_T_K'] = abs(res['t0_hat_K'] - truth_boundary[1])
    else:
        res.setdefault('boundary_err_A', None)
        res.setdefault('boundary_err_T_K', None)
    return res


def fmt_cpd(res):
    if res['mask'] is None:
        return "  вырожденный случай: шаг не выделен (p=%.3f)" % res['p_value']
    lines = [
        "  шаг: %s, граница A>=%.2f, T>=%.0f K (beta=%+.3f, R2=%.2f)"
        % (res['orientation'], res['a0_hat'], res['t0_hat_K'],
           res['beta_hat'], res['r2_gain']),
        "  аномалий: %d | зашитых точек: %d" % (res['n_anomalies'], res['n_planted']),
        "  precision=%.2f recall=%.2f F1=%.2f  p(пермут.)=%.4f"
        % (res['precision'], res['recall'], res['f1'], res['p_value']),
    ]
    if res['boundary_err_A'] is not None:
        lines.append("  ошибка границы: ΔA=%.3f, ΔT=%.0f K"
                     % (res['boundary_err_A'], res['boundary_err_T_K']))
    lines.append("  Вердикт: %s" % ("ОБНАРУЖЕНО" if res['detected'] else "НЕ ОБНАРУЖЕНО"))
    return "\n".join(lines)
