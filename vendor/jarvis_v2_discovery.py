"""
Тест «автономного открытия скрытой физики».

Оригинальный ноутбук заявлял способность находить фазовый переход, но за все
прогоны выдавал 0 новых гипотез. Здесь — проверяемая версия:

  1. BO собирает данные на ландшафте со СКРЫТЫМ разрывом.
  2. Обучается GP, считаются остатки.
  3. Два варианта аномалий:
       mode='abs'     — |остаток| > median + k·MAD
       mode='signed'  — положительный остаток > median + k·MAD
                        (переход добавляет +0.9, поэтому остатки должны
                         быть систематически положительными)
  4. Аномалии сверяются с ЗАШИТОЙ истиной (A > 0.30 ∧ T > 550 K)
     через precision / recall / F1.
"""
import numpy as np


def _prep_X(X):
    """(A, B, C, T) -> признаки GP с нормировкой температуры в [0,1]."""
    X = np.asarray(X, dtype=float)
    t = (X[:, 3] - 300.0) / 700.0
    return np.column_stack([X[:, 0], X[:, 1], X[:, 2], t])


def _fit_gp(X, y):
    """GP-суррогат (тот же конфиг, что у BO-методов). Возвращает функцию
    предсказания mu(X_new). Единственное место с GP-конфигом детекции —
    используется и discover_phase_boundary, и jarvis_v2_cpd."""
    from sklearn.gaussian_process import GaussianProcessRegressor
    from sklearn.gaussian_process.kernels import Matern, ConstantKernel, WhiteKernel
    from sklearn.preprocessing import StandardScaler

    Xs_ = _prep_X(X)
    scaler = StandardScaler().fit(Xs_)
    gp = GaussianProcessRegressor(
        kernel=ConstantKernel(1.0) * Matern(nu=2.5)
                + WhiteKernel(0.02, noise_level_bounds=(1e-10, 1.0)),
        alpha=1e-6, normalize_y=True, n_restarts_optimizer=1, random_state=0,
    ).fit(scaler.transform(Xs_), np.asarray(y, dtype=float))
    return lambda X_new: gp.predict(scaler.transform(_prep_X(X_new)))


def discover_phase_boundary(X, y, truth_fn, k_mad=3.0, min_anom=3, mode='abs'):
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float)

    mu = _fit_gp(X, y)(X)
    resid = np.abs(y - mu) if mode == 'abs' else (y - mu)

    med = np.median(resid)
    mad = np.median(np.abs(resid - med))
    thr = med + k_mad * (mad + 1e-12)
    anom = resid > thr

    planted = truth_fn(X[:, 0], X[:, 1], X[:, 2], X[:, 3])

    tp = int(np.sum(anom & planted))
    fp = int(np.sum(anom & ~planted))
    fn = int(np.sum(~anom & planted))
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0

    return {
        'n': len(y),
        'n_anomalies': int(anom.sum()),
        'n_planted': int(planted.sum()),
        'threshold': float(thr),
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'anom_A': (float(np.min(X[anom, 0])), float(np.max(X[anom, 0]))) if anom.any() else None,
        'anom_T': (float(np.min(X[anom, 3])), float(np.max(X[anom, 3]))) if anom.any() else None,
        'enough': anom.sum() >= min_anom,
        'X': X, 'y': y, 'resid': resid, 'anom': anom, 'planted': planted,
    }


def fmt_discovery(res, mode='abs'):
    lines = []
    lines.append("  Экспериментов: %d | аномалий: %d | точек в зашитой области: %d"
                 % (res['n'], res['n_anomalies'], res['n_planted']))
    lines.append("  Порог (median + 3·MAD): %.4f" % res['threshold'])
    if res['anom_A'] is not None:
        lines.append("  Область аномалий: A ∈ [%.2f, %.2f], T ∈ [%.0f, %.0f] K"
                     % (res['anom_A'][0], res['anom_A'][1],
                        res['anom_T'][0], res['anom_T'][1]))
    lines.append("  Сверка с зашитой истиной (A>0.30 ∧ T>550K):")
    lines.append("    precision=%.2f  recall=%.2f  F1=%.2f"
                 % (res['precision'], res['recall'], res['f1']))
    verdict = ("ОБНАРУЖЕНО" if res['f1'] >= 0.5 and res['n_anomalies'] >= 3
               else "НЕ ОБНАРУЖЕНО")
    lines.append("  Вердикт: %s" % verdict)
    return "\n".join(lines)