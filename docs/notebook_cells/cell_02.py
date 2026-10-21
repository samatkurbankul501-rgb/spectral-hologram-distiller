import time
import numpy as np
from scipy import linalg
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error, r2_score


class HologramCrystalLoader:
    """
    Клиентский SDK для работы с файлами .hologram.npz:
    1. Мгновенная загрузка кристалла (X, y, gamma_weights) для любых ML/DL моделей
    2. Загрузка срезов (Anytime-обучение на первых k точках)
    3. Радар сдвига домена (Subspace Bures-Wasserstein OOD Detector) по 16-битному паспорту
    """
    def __init__(self, filepath: str = "reactor_crystal_v33.hologram.npz"):
        data = np.load(filepath)
        self.X = data["crystal_X"].astype(np.float64)
        self.y = data["crystal_Y"].astype(np.float64)
        self.weights = data["gamma_weights"].astype(np.float64)
        self.eigvals = data["passport_eigvals"].astype(np.float64)
        self.eigvecs = data["passport_eigvecs"].astype(np.float64)
        self.X_mean = data["X_mean"].astype(np.float64)
        self.X_std = data["X_std"].astype(np.float64)
        self.n_seen = int(data["n_seen"][0])

        # Реконструируем спектральный оператор Psi_passport из топ-32 собственных мод
        self.Psi_passport = (self.eigvecs * self.eigvals) @ self.eigvecs.T

    def get_training_data(self, budget_points: int = None):
        """Возвращает (X, y, sample_weight) для выбранного бюджета точек (Anytime-режим)."""
        if budget_points is None or budget_points >= len(self.X):
            return self.X, self.y, self.weights
        w_sub = self.weights[:budget_points].copy()
        w_sub /= np.mean(w_sub)
        return self.X[:budget_points], self.y[:budget_points], w_sub

    def check_domain_shift(self, distiller_ref, X_new_stream: np.ndarray) -> dict:
        """
        Сверяет новый поток данных со Спектральным Паспортом через проекцию на 32 главные моды.
        Работает за ~2 мс без обращения к исходному датасету!
        """
        t0 = time.perf_counter()
        U_new = distiller_ref._spectral_transform(X_new_stream)
        Psi_new = (U_new.T @ U_new) / len(U_new)

        # Проектируем новый спектр в 32-мерное собственное подпространство паспорта (32x32 вместо 256x256!)
        V = self.eigvecs # (256, 32)
        A_sub = np.diag(np.maximum(self.eigvals, 1e-7))
        B_sub = V.T @ Psi_new @ V + 1e-7 * np.eye(len(self.eigvals))
        B_sub = 0.5 * (B_sub + B_sub.T)

        sqrt_A = np.diag(np.sqrt(np.diag(A_sub)))
        inner = sqrt_A @ B_sub @ sqrt_A
        inner = 0.5 * (inner + inner.T)
        eig_inner = np.maximum(linalg.eigvalsh(inner), 0.0)

        bw_dist = float(np.sqrt(max(0.0, np.trace(A_sub) + np.trace(B_sub) - 2.0 * np.sum(np.sqrt(eig_inner)))))
        latency_ms = (time.perf_counter() - t0) * 1000.0

        return {
            "bw_distance": bw_dist,
            "latency_ms": latency_ms,
            "is_shifted": bw_dist > 0.085
        }


# --- 1. ТЕСТ УНИВЕРСАЛЬНОСТИ КРИСТАЛЛА (NEURAL NET & GRADIENT BOOSTING) ---
loader = HologramCrystalLoader("reactor_crystal_v33.hologram.npz")
X_holo, y_holo, w_holo = loader.get_training_data()

print("=" * 85)
print("ТЕСТ 1: МЕЖМОДЕЛЬНАЯ УНИВЕРСАЛЬНОСТЬ КРИСТАЛЛА (21.1 КБ -> Градиентный Бустинг и Нейросеть)")
print("=" * 85)

# Сравним обучение тяжелого Градиентного Бустинга и Нейросети (MLP) на Random vs K-Means vs Hologram
cross_models = {
    "HistGradientBoosting (Бустинг)": lambda: HistGradientBoostingRegressor(
        max_iter=120, max_leaf_nodes=15, min_samples_leaf=3, l2_regularization=0.1, random_state=42
    ),
    "MLP Neural Net (Нейросеть)": lambda: MLPRegressor(
        hidden_layer_sizes=(64, 64), activation="tanh", alpha=0.02, max_iter=600, random_state=42
    )
}

for model_title, builder in cross_models.items():
    print(f"\n>>> Архитектура: {model_title}")
    print(f"{'Выборка (200 точек = 0.2%)':<28} | {'MSE (Норма)':<14} | {'MSE (Фазовый пик)':<18} | {'R² (Фазовый пик)'}")
    print("-" * 85)

    for ds_name, (X_tr, y_tr, w_tr) in [
        ("Random Coreset (0.2%)", (X_rand, y_rand, None)),
        ("K-Means Coreset (0.2%)", (X_km, y_km, km_weights)),
        ("Spectral Hologram v3.3", (X_holo, y_holo, w_holo))
    ]:
        reg = builder()
        # MLPRegressor в старых версиях sklearn не принимает sample_weight напрямую, делаем ресэмплинг по весам
        if "MLP" in model_title and w_tr is not None:
            rng_mlp = np.random.default_rng(42)
            probs = w_tr / np.sum(w_tr)
            boot_idx = rng_mlp.choice(len(X_tr), size=len(X_tr) * 2, replace=True, p=probs)
            reg.fit(X_tr[boot_idx], y_tr[boot_idx])
        elif "MLP" in model_title:
            reg.fit(X_tr, y_tr)
        else:
            reg.fit(X_tr, y_tr, sample_weight=w_tr)

        mse_n = mean_squared_error(y_test_norm, reg.predict(X_test_norm))
        pred_c = reg.predict(X_test_crit)
        mse_c = mean_squared_error(y_test_crit, pred_c)
        r2_c = r2_score(y_test_crit, pred_c)
        print(f"{ds_name:<28} | {mse_n:<14.4f} | {mse_c:<18.4f} | {r2_c:.4f}")


# --- 2. ТЕСТ РАДАРА СДВИГА ДОМЕНА (BURES-WASSERSTEIN OOD DETECTOR НА 16-БИТНОМ ПАСПОРТЕ) ---
print("\n" + "=" * 85)
print("ТЕСТ 2: РАДАР СДВИГА ДОМЕНА ПО 16-БИТНОМУ СПЕКТРАЛЬНОМУ ПАСПОРТУ (БЕЗ ИСХОДНЫХ ДАННЫХ)")
print("=" * 85)

# Поток 1: Штатный режим работы реактора
X_stream_ok, _, _ = next(stream_reactor_dataset_4d(n_samples=2000, batch_size=2000, seed=555))
res_ok = loader.check_domain_shift(distiller, X_stream_ok)

# Поток 2: Разгерметизация / дрейф датчиков давления и концентрации
X_stream_drift = X_stream_ok.copy()
X_stream_drift[:, 0:2] = X_stream_drift[:, 0:2] * 1.65 + 1.25
res_drift = loader.check_domain_shift(distiller, X_stream_drift)

print(f"Штатный поток (2000 строк):   Дистанция BW = {res_ok['bw_distance']:.4f} | Время = {res_ok['latency_ms']:.2f} мс | Статус: {'ТРЕВОГА' if res_ok['is_shifted'] else 'НОРМА (OK)'}")
print(f"Аварийный сдвиг (2000 строк): Дистанция BW = {res_drift['bw_distance']:.4f} | Время = {res_drift['latency_ms']:.2f} мс | Статус: {'ТРЕВОГА (DOMAIN SHIFT!)' if res_drift['is_shifted'] else 'НОРМА'}")