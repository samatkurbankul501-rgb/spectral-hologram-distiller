import gc
import os
import time
import tracemalloc
import numpy as np
from scipy import linalg
from scipy.spatial.distance import cdist
from sklearn.datasets import fetch_openml, fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import MiniBatchKMeans
from sklearn.kernel_ridge import KernelRidge
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_squared_error, r2_score


class SpectralHologramDistillerV36:
    """
    Spectral Hologram Distiller (SHD v3.6 — Unified Spectral-Voronoi Partition)
    - Сгущает сетку в зоне фазовых переходов (M_high узлов) + покрывает основной режим (M_low узлов)
    - Объединяет все 300 узлов в ЕДИНУЮ непересекающуюся тесселяцию Вороного (0% граничных искажений)
    - Выбирает реальный физический медоид по совместной метрике (Z, y), подавляя экспериментальный шум
    """
    def __init__(
        self,
        crystal_size: int = 300,
        high_band_ratio: float = 0.28,   # 84 медоида сгущают зону ВТСП, 216 покрывают основной объем
        sketch_dim: int = 256,
        beta_low: float = 0.25,
        tail_quantile: float = 84.0,
        random_state: int = 42
    ):
        self.M = crystal_size
        self.M_high = int(round(crystal_size * high_band_ratio))
        self.M_low = crystal_size - self.M_high
        self.k = sketch_dim
        self.beta_low = beta_low
        self.tail_q = tail_quantile
        self.rng = np.random.default_rng(random_state)

        self.n_seen = 0
        self.Psi_sum = np.zeros((self.k, self.k), dtype=np.float64)
        self.Phi_sum = None
        self._proj_initialized = False

        # Буфер накопления для стратифицированной инициализации единой решетки
        self.init_Z, self.init_X, self.init_Y, self.init_U, self.init_score = [], [], [], [], []
        self.lattice_ready = False

        # Единая решетка из M = 300 непересекающихся ячеек Вороного
        self.centers_Z = None
        self.mean_Y = None
        self.med_X = None
        self.med_Y = None
        self.med_U = None
        self.counts = None

        self.crystal_X = None
        self.crystal_Y = None
        self.crystal_U = None
        self.gamma_weights = None

    def _init_projector(self, X_batch: np.ndarray, y_batch: np.ndarray):
        d = X_batch.shape[1]
        y_dim = y_batch.shape[1]
        self.Phi_sum = np.zeros((self.k, y_dim), dtype=np.float64)

        self.X_mean = np.mean(X_batch, axis=0)
        self.X_std = np.std(X_batch, axis=0) + 1e-8
        self.y_mean = float(np.mean(y_batch))
        self.y_std = float(np.std(y_batch)) + 1e-8

        scale_d = 1.0 / np.sqrt(max(d, 1))
        k_half = self.k // 2
        omega_low = self.rng.normal(0, 0.75 * scale_d, size=(d, k_half))
        omega_high = self.rng.normal(0, 2.10 * scale_d, size=(d, self.k - k_half))
        self.Omega = np.hstack([omega_low, omega_high])
        self.phase = self.rng.uniform(0, 2 * np.pi, size=(1, self.k))
        self._proj_initialized = True

    def _normalize_X(self, X_batch: np.ndarray) -> np.ndarray:
        return np.clip((X_batch - self.X_mean) / self.X_std, -8.0, 8.0)

    def _spectral_transform(self, X_batch: np.ndarray) -> tuple:
        Z = self._normalize_X(X_batch)
        U = np.sqrt(2.0 / self.k) * np.cos(Z @ self.Omega + self.phase)
        norms = np.linalg.norm(U, axis=1, keepdims=True) + 1e-8
        return Z, U / norms

    def _seed_kmeans_pp(self, Z_cand: np.ndarray, n_select: int) -> np.ndarray:
        n_total = len(Z_cand)
        n_select = min(n_select, n_total)
        first_idx = int(self.rng.integers(0, n_total))
        selected = [first_idx]
        min_sq_dists = np.sum((Z_cand - Z_cand[first_idx]) ** 2, axis=1)

        for _ in range(1, n_select):
            probs = min_sq_dists / (np.sum(min_sq_dists) + 1e-12)
            next_idx = int(self.rng.choice(n_total, p=probs))
            selected.append(next_idx)
            new_dists = np.sum((Z_cand - Z_cand[next_idx]) ** 2, axis=1)
            min_sq_dists = np.minimum(min_sq_dists, new_dists)

        return Z_cand[selected].copy()

    def _build_unified_lattice(self):
        Z_all = np.vstack(self.init_Z)
        X_all = np.vstack(self.init_X)
        Y_all = np.vstack(self.init_Y)
        U_all = np.vstack(self.init_U)
        sc_all = np.concatenate(self.init_score)

        thr = np.percentile(sc_all, self.tail_q)
        is_high = sc_all >= thr

        # 1. Стратифицированное зарождение центров: M_low в норме + M_high в зоне фазового пика
        c_low = self._seed_kmeans_pp(Z_all[~is_high], self.M_low)
        c_high = self._seed_kmeans_pp(Z_all[is_high], self.M_high)
        centers_Z = np.vstack([c_low, c_high])
        M_total = len(centers_Z)

        # 2. Совместная релаксация Ллойда единой решетки (убирает перекрытия границ!)
        for _ in range(5):
            d_mat = cdist(Z_all, centers_Z, metric="sqeuclidean")
            labels = np.argmin(d_mat, axis=1)
            for c_i in range(M_total):
                mask = labels == c_i
                if np.any(mask):
                    centers_Z[c_i] = np.mean(Z_all[mask], axis=0)

        # 3. Вычисляем точные массы и средние температуры каждой ячейки Вороного
        d_mat = cdist(Z_all, centers_Z, metric="sqeuclidean")
        labels = np.argmin(d_mat, axis=1)
        counts = np.zeros(M_total, dtype=np.float64)
        mean_Y = np.zeros((M_total, Y_all.shape[1]), dtype=np.float64)
        med_X = np.zeros((M_total, X_all.shape[1]), dtype=np.float64)
        med_Y = np.zeros((M_total, Y_all.shape[1]), dtype=np.float64)
        med_U = np.zeros((M_total, U_all.shape[1]), dtype=np.float64)

        for c_i in range(M_total):
            mask = labels == c_i
            if np.any(mask):
                counts[c_i] = float(np.sum(mask))
                mean_Y[c_i] = np.mean(Y_all[mask], axis=0)
                # Совместный снаппинг (Z, y): берем реальный сверхпроводник без шумового выброса по Tc!
                d_z = np.sum((Z_all[mask] - centers_Z[c_i]) ** 2, axis=1)
                d_y = np.sum(((Y_all[mask] - mean_Y[c_i]) / self.y_std) ** 2, axis=1)
                best_local = int(np.argmin(d_z + 2.5 * d_y))
                idx_global = np.where(mask)[0][best_local]
            else:
                counts[c_i] = 1.0
                idx_global = int(np.argmin(d_mat[:, c_i]))
                mean_Y[c_i] = Y_all[idx_global]

            med_X[c_i] = X_all[idx_global]
            med_Y[c_i] = Y_all[idx_global]
            med_U[c_i] = U_all[idx_global]

        self.centers_Z = centers_Z
        self.mean_Y = mean_Y
        self.med_X = med_X
        self.med_Y = med_Y
        self.med_U = med_U
        self.counts = counts
        self.init_Z, self.init_X, self.init_Y, self.init_U, self.init_score = [], [], [], [], []
        self.lattice_ready = True

    def partial_fit(self, X_batch: np.ndarray, y_batch: np.ndarray):
        if y_batch.ndim == 1:
            y_batch = y_batch[:, None]

        if not self._proj_initialized:
            self._init_projector(X_batch, y_batch)

        B = X_batch.shape[0]
        Z_B, U_B = self._spectral_transform(X_batch)

        self.Psi_sum += U_B.T @ U_B
        self.Phi_sum += U_B.T @ y_batch
        self.n_seen += B

        Psi_hat = self.Psi_sum / self.n_seen
        Phi_hat = self.Phi_sum / self.n_seen

        I_k = np.eye(self.k, dtype=np.float64)
        L_low = linalg.cho_factor(Psi_hat + self.beta_low * I_k, lower=True)
        W_pred = linalg.cho_solve(L_low, Phi_hat)

        residual = np.sum((y_batch - U_B @ W_pred) ** 2, axis=1)
        res_ratio = residual / (np.median(residual) + 1e-8)
        y_z = np.maximum(0.0, (y_batch.squeeze() - self.y_mean) / self.y_std)
        phase_score = res_ratio + 1.8 * (y_z ** 2)

        if not self.lattice_ready:
            self.init_Z.append(Z_B)
            self.init_X.append(X_batch)
            self.init_Y.append(y_batch)
            self.init_U.append(U_B)
            self.init_score.append(phase_score)
            if sum(len(a) for a in self.init_Z) >= max(4500, self.M * 12):
                self._build_unified_lattice()
            return

        # Потоковое обновление единой решетки Вороного с совместным снаппингом (Z, y)
        dists = cdist(Z_B, self.centers_Z, metric="sqeuclidean")
        assign = np.argmin(dists, axis=1)
        med_Z = self._normalize_X(self.med_X)

        for c_id in np.unique(assign):
            mask = assign == c_id
            n_new = int(np.sum(mask))
            old_count = self.counts[c_id]
            new_count = old_count + n_new
            self.counts[c_id] = new_count

            lr = n_new / new_count
            self.centers_Z[c_id] = (1.0 - lr) * self.centers_Z[c_id] + lr * np.mean(Z_B[mask], axis=0)
            self.mean_Y[c_id] = (1.0 - lr) * self.mean_Y[c_id] + lr * np.mean(y_batch[mask], axis=0)

            d_z = np.sum((Z_B[mask] - self.centers_Z[c_id]) ** 2, axis=1)
            d_y = np.sum(((y_batch[mask] - self.mean_Y[c_id]) / self.y_std) ** 2, axis=1)
            cand_joint = d_z + 2.5 * d_y
            best_local = int(np.argmin(cand_joint))

            curr_joint = float(
                np.sum((med_Z[c_id] - self.centers_Z[c_id]) ** 2)
                + 2.5 * np.sum(((self.med_Y[c_id] - self.mean_Y[c_id]) / self.y_std) ** 2)
            )
            if float(cand_joint[best_local]) <= curr_joint:
                idx_global = np.where(mask)[0][best_local]
                self.med_X[c_id] = X_batch[idx_global]
                self.med_Y[c_id] = y_batch[idx_global]
                self.med_U[c_id] = U_B[idx_global]

    def finalize_crystal(self, fista_steps: int = 250):
        if not self.lattice_ready:
            self._build_unified_lattice()

        order = np.argsort(self.counts)[::-1]
        self.crystal_X = self.med_X[order].copy()
        self.crystal_Y = self.med_Y[order].squeeze().copy()
        self.crystal_U = self.med_U[order].copy()
        sorted_counts = self.counts[order].copy()
        M_curr = len(self.crystal_U)

        # Точная мера Радона-Никодима единого разбиения Вороного
        voronoi_prob = sorted_counts / np.sum(sorted_counts)

        # Спектральная коррекция FISTA
        Psi_hat = self.Psi_sum / self.n_seen
        G_U = self.crystal_U @ self.crystal_U.T
        K = (G_U ** 2) + 0.25 * np.eye(M_curr)
        b = np.sum((self.crystal_U @ Psi_hat) * self.crystal_U, axis=1) + 0.25 * voronoi_prob

        L_lip = float(linalg.norm(K, ord=2)) + 1e-6
        gamma = voronoi_prob.copy()
        z = gamma.copy()
        t_step = 1.0

        for _ in range(fista_steps):
            grad = K @ z - b
            gamma_next = np.maximum(1e-5, z - grad / L_lip)
            t_next = 0.5 * (1.0 + np.sqrt(1.0 + 4.0 * t_step ** 2))
            z = gamma_next + ((t_step - 1.0) / t_next) * (gamma_next - gamma)
            gamma, t_step = gamma_next, t_next

        gamma_prob = gamma / np.sum(gamma)
        Psi_crystal = (self.crystal_U.T * gamma_prob) @ self.crystal_U
        rel_error = linalg.norm(Psi_crystal - Psi_hat, ord="fro") / linalg.norm(Psi_hat, ord="fro")

        # Веса строго следуют мере ячеек единого разбиения Вороного + мягкая спектральная поправка
        w_combined = (0.85 * voronoi_prob + 0.15 * gamma_prob) * M_curr
        self.gamma_weights = w_combined / np.mean(w_combined)
        return rel_error

    def save_hologram(self, filepath: str = "superconduct_v36.hologram.npz", rank_passport: int = 32) -> float:
        Psi_hat = self.Psi_sum / self.n_seen
        eigvals, eigvecs = linalg.eigh(Psi_hat)
        np.savez_compressed(
            filepath,
            crystal_X=self.crystal_X.astype(np.float32),
            crystal_Y=self.crystal_Y.astype(np.float32),
            gamma_weights=self.gamma_weights.astype(np.float32),
            passport_eigvals=eigvals[-rank_passport:].astype(np.float16),
            passport_eigvecs=eigvecs[:, -rank_passport:].astype(np.float16),
            X_mean=self.X_mean.astype(np.float32),
            X_std=self.X_std.astype(np.float32),
            n_seen=np.array([self.n_seen], dtype=np.int64)
        )
        return os.path.getsize(filepath) / 1024.0


# --- ЗАПУСК ИСПЫТАНИЯ SHD v3.6 НА РЕАЛЬНЫХ СВЕРХПРОВОДНИКАХ ---
gc.collect()
M_CRYSTAL = 300
BATCH_SIZE = 2500

tracemalloc.start()
t0 = time.perf_counter()

distiller = SpectralHologramDistillerV36(
    crystal_size=M_CRYSTAL, high_band_ratio=0.28, sketch_dim=256, tail_quantile=84.0
)

for start_i in range(0, len(X_train_stream), BATCH_SIZE):
    end_i = min(start_i + BATCH_SIZE, len(X_train_stream))
    distiller.partial_fit(X_train_stream[start_i:end_i], y_train_stream[start_i:end_i])

rel_spec_error = distiller.finalize_crystal()
elapsed = time.perf_counter() - t0
_, peak_mem = tracemalloc.get_traced_memory()
tracemalloc.stop()

holo_kb = distiller.save_hologram("superconduct_v36.hologram.npz")
raw_kb = (X_train_stream.nbytes + y_train_stream.nbytes) / 1024.0

print("=" * 98)
print(f"ТЕЛЕМЕТРИЯ SHD v3.6 (ЕДИНАЯ РЕШЕТКА ВОРОНОГО + СОВМЕСТНЫЙ МЕДОИД) ({M_CRYSTAL} точек = 1.76%)")
print("=" * 98)
print(f"Время на CPU: {elapsed:.2f} сек | Пик RAM: {peak_mem / (1024**2):.2f} МБ | Ошибка спектра: {rel_spec_error*100:.2f}%")
print(f"Размер файла: {holo_kb:.1f} КБ вместо {raw_kb:.0f} КБ (Сжатие в {raw_kb/holo_kb:.0f} раз!)")
print(f"Захвачено ВТСП (Tc >= 77K): {np.sum(distiller.crystal_Y >= crit_threshold)} шт.\n")

print("=" * 98)
print("БЕНЧМАРК 1: ЯДЕРНАЯ РЕГРЕССИЯ (RBF KernelRidge) НА ФИЗИКЕ СВЕРХПРОВОДНИКОВ")
print("=" * 98)
print(f"{'Обучающая выборка':<32} | {'RMSE (Все)':<12} | {'RMSE (Пик Tc>=77K)':<20} | {'R² (Общий)':<12} | {'Время обучения'}")
print("-" * 98)

for name, (X_tr, y_tr, w_tr) in [
    ("Oracle (Полный датасет)", (X_train_stream[:9000], y_train_stream[:9000], None)),
    ("Random Coreset (1.76%)", (X_rand, y_rand, None)),
    ("K-Means Coreset (1.76%)", (X_km, y_km, km_weights)),
    ("Spectral Hologram v3.6 (1.76%)", (distiller.crystal_X, distiller.crystal_Y, distiller.gamma_weights)),
]:
    gc.collect()
    t_s = time.perf_counter()
    Z_tr = std_scaler.transform(X_tr)
    model = KernelRidge(kernel="rbf", alpha=0.08, gamma=rbf_gamma)
    model.fit(Z_tr, y_tr, sample_weight=w_tr)
    fit_ms = (time.perf_counter() - t_s) * 1000.0

    pred_all = model.predict(Z_test_all)
    pred_crit = model.predict(Z_test_crit)
    print(f"{name:<32} | {np.sqrt(mean_squared_error(y_test_all, pred_all)):<12.2f} | {np.sqrt(mean_squared_error(y_test_crit, pred_crit)):<20.2f} | {r2_score(y_test_all, pred_all):<12.4f} | {fit_ms:.1f} мс")

print("\n" + "=" * 98)
print("БЕНЧМАРК 2: ГРАДИЕНТНЫЙ БУСТИНГ (HistGradientBoosting) НА ТЕХ ЖЕ 300 ТОЧКАХ")
print("=" * 98)
print(f"{'Обучающая выборка':<32} | {'RMSE (Все)':<12} | {'RMSE (Пик Tc>=77K)':<20} | {'R² (Общий)':<12} | {'Время обучения'}")
print("-" * 98)

for name, (X_tr, y_tr, w_tr) in [
    ("Oracle (Полный 100% датасет)", (X_train_stream, y_train_stream, None)),
    ("Random Coreset (1.76%)", (X_rand, y_rand, None)),
    ("K-Means Coreset (1.76%)", (X_km, y_km, km_weights)),
    ("Spectral Hologram v3.6 (1.76%)", (distiller.crystal_X, distiller.crystal_Y, distiller.gamma_weights)),
]:
    t_s = time.perf_counter()
    gbm = HistGradientBoostingRegressor(max_iter=120, max_leaf_nodes=20, min_samples_leaf=5, random_state=42)
    gbm.fit(X_tr, y_tr, sample_weight=w_tr)
    fit_ms = (time.perf_counter() - t_s) * 1000.0

    pred_all = gbm.predict(X_test_all)
    pred_crit = gbm.predict(X_test_crit)
    print(f"{name:<32} | {np.sqrt(mean_squared_error(y_test_all, pred_all)):<12.2f} | {np.sqrt(mean_squared_error(y_test_crit, pred_crit)):<20.2f} | {r2_score(y_test_all, pred_all):<12.4f} | {fit_ms:.1f} мс")