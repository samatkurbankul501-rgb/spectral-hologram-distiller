import gc
import os
import time
import tracemalloc
import numpy as np
import matplotlib.pyplot as plt
from scipy import linalg
from scipy.spatial.distance import cdist
from sklearn.datasets import fetch_openml, fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import MiniBatchKMeans
from sklearn.kernel_ridge import KernelRidge
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_squared_error, r2_score


# --- 1. ЯДРО SPECTRAL HOLOGRAM DISTILLER v3.5 (RADON-NIKODYM CALIBRATED) ---
class SpectralHologramDistillerV35:
    """
    Spectral Hologram Distiller (SHD v3.5 — Radon-Nikodym Density-Calibrated)
    - Сохраняет 2-кратное сгущение медоидов в зоне фазового пика (высокое разрешение ВТСП)
    - Взвешивает полосы по теореме Радона-Никодима (N_band / M_band), устраняя смещение на общем потоке!
    """
    def __init__(
        self,
        crystal_size: int = 300,
        high_band_ratio: float = 0.30,   # 90 медоидов на ВТСП-пик, 210 медоидов на основной закон
        sketch_dim: int = 256,
        beta_low: float = 0.25,
        tail_quantile: float = 85.0,
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
        self.n_low_seen = 0
        self.n_high_seen = 0
        self.Psi_sum = np.zeros((self.k, self.k), dtype=np.float64)
        self.Phi_sum = None
        self._proj_initialized = False

        # Контур А (Backbone — 210 медоидов)
        self.low_centers_Z = None
        self.low_X = None
        self.low_Y = None
        self.low_U = None
        self.low_counts = None

        # Контур Б (Phase-Peak — 90 медоидов)
        self.high_pool_Z, self.high_pool_X, self.high_pool_Y, self.high_pool_U = [], [], [], []
        self.high_centers_Z = None
        self.high_X = None
        self.high_Y = None
        self.high_U = None
        self.high_counts = None

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

    def _init_voronoi_lattice(
        self, Z_cand: np.ndarray, X_cand: np.ndarray, y_cand: np.ndarray, U_cand: np.ndarray, n_select: int
    ):
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

        centers_Z = Z_cand[selected].copy()

        for _ in range(4):
            d_mat = cdist(Z_cand, centers_Z, metric="sqeuclidean")
            labels = np.argmin(d_mat, axis=1)
            for c_i in range(n_select):
                mask = labels == c_i
                if np.any(mask):
                    centers_Z[c_i] = np.mean(Z_cand[mask], axis=0)

        d_final = cdist(centers_Z, Z_cand, metric="sqeuclidean")
        med_idx = np.argmin(d_final, axis=1)
        counts = np.bincount(labels, minlength=n_select).astype(np.float64) + 1.0

        return (
            centers_Z,
            X_cand[med_idx].copy(),
            y_cand[med_idx].copy(),
            U_cand[med_idx].copy(),
            counts,
        )

    @staticmethod
    def _update_voronoi_medoids(Z_b, X_b, y_b, U_b, centers_Z, med_X, med_Y, med_U, counts, X_mean, X_std):
        if len(Z_b) == 0:
            return
        dists = cdist(Z_b, centers_Z, metric="sqeuclidean")
        assign = np.argmin(dists, axis=1)
        med_Z = np.clip((med_X - X_mean) / X_std, -8.0, 8.0)

        for c_id in np.unique(assign):
            mask = assign == c_id
            n_new = int(np.sum(mask))
            counts[c_id] += n_new
            lr = min(0.25, n_new / counts[c_id])
            centers_Z[c_id] = (1.0 - lr) * centers_Z[c_id] + lr * np.mean(Z_b[mask], axis=0)

            d_to_center = np.sum((Z_b[mask] - centers_Z[c_id]) ** 2, axis=1)
            best_local = int(np.argmin(d_to_center))
            cand_dist = float(d_to_center[best_local])
            curr_dist = float(np.sum((med_Z[c_id] - centers_Z[c_id]) ** 2))

            if cand_dist <= curr_dist:
                idx_global = np.where(mask)[0][best_local]
                med_X[c_id] = X_b[idx_global]
                med_Y[c_id] = y_b[idx_global]
                med_U[c_id] = U_b[idx_global]

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
        thr = np.percentile(phase_score, self.tail_q)
        is_high_phase = phase_score >= thr

        # Считаем реальную массу потока в каждой полосе для коррекции Радона-Никодима
        self.n_low_seen += int(np.sum(~is_high_phase))
        self.n_high_seen += int(np.sum(is_high_phase))

        # --- КОНТУР А: Основное многообразие (Backbone) ---
        Z_norm = Z_B[~is_high_phase]
        X_norm = X_batch[~is_high_phase]
        y_norm = y_batch[~is_high_phase]
        U_norm = U_B[~is_high_phase]

        if self.low_X is None and len(Z_norm) >= self.M_low:
            (
                self.low_centers_Z,
                self.low_X,
                self.low_Y,
                self.low_U,
                self.low_counts,
            ) = self._init_voronoi_lattice(Z_norm, X_norm, y_norm, U_norm, self.M_low)
        elif self.low_X is not None:
            self._update_voronoi_medoids(
                Z_norm, X_norm, y_norm, U_norm,
                self.low_centers_Z, self.low_X, self.low_Y, self.low_U, self.low_counts,
                self.X_mean, self.X_std
            )

        # --- КОНТУР Б: Высокотемпературный / Нелинейный фазовый пик ---
        Z_sp = Z_B[is_high_phase]
        X_sp = X_batch[is_high_phase]
        y_sp = y_batch[is_high_phase]
        U_sp = U_B[is_high_phase]

        if self.high_X is None:
            if len(Z_sp) > 0:
                self.high_pool_Z.append(Z_sp)
                self.high_pool_X.append(X_sp)
                self.high_pool_Y.append(y_sp)
                self.high_pool_U.append(U_sp)
                total_pooled = sum(len(arr) for arr in self.high_pool_Z)
                if total_pooled >= self.M_high * 2:
                    Zp = np.vstack(self.high_pool_Z)
                    Xp = np.vstack(self.high_pool_X)
                    Yp = np.vstack(self.high_pool_Y)
                    Up = np.vstack(self.high_pool_U)
                    (
                        self.high_centers_Z,
                        self.high_X,
                        self.high_Y,
                        self.high_U,
                        self.high_counts,
                    ) = self._init_voronoi_lattice(Zp, Xp, Yp, Up, self.M_high)
                    self.high_pool_Z, self.high_pool_X, self.high_pool_Y, self.high_pool_U = [], [], [], []
        else:
            self._update_voronoi_medoids(
                Z_sp, X_sp, y_sp, U_sp,
                self.high_centers_Z, self.high_X, self.high_Y, self.high_U, self.high_counts,
                self.X_mean, self.X_std
            )

    def finalize_crystal(self, fista_steps: int = 300):
        if self.high_X is None and len(self.high_pool_Z) > 0:
            Zp = np.vstack(self.high_pool_Z)
            Xp = np.vstack(self.high_pool_X)
            Yp = np.vstack(self.high_pool_Y)
            Up = np.vstack(self.high_pool_U)
            (
                self.high_centers_Z,
                self.high_X,
                self.high_Y,
                self.high_U,
                self.high_counts,
            ) = self._init_voronoi_lattice(Zp, Xp, Yp, Up, self.M_high)

        low_order = np.argsort(self.low_counts)[::-1]
        high_order = np.argsort(self.high_counts)[::-1]

        X_l, Y_l, U_l, C_l = self.low_X[low_order], self.low_Y[low_order], self.low_U[low_order], self.low_counts[low_order]
        X_h, Y_h, U_h, C_h = self.high_X[high_order], self.high_Y[high_order], self.high_U[high_order], self.high_counts[high_order]

        # Закон Радона-Никодима: вес каждого медоида пропорционален массе его ячейки Вороного
        # и глобальной доле его фазовой полосы в потоке!
        w_l_prior = (C_l / np.sum(C_l)) * (self.n_low_seen / self.n_seen)
        w_h_prior = (C_h / np.sum(C_h)) * (self.n_high_seen / self.n_seen)

        comb_X, comb_Y, comb_U, comb_prior = [], [], [], []
        il, ih = 0, 0
        while il < len(X_l) or ih < len(X_h):
            for _ in range(7):
                if il < len(X_l):
                    comb_X.append(X_l[il])
                    comb_Y.append(Y_l[il])
                    comb_U.append(U_l[il])
                    comb_prior.append(w_l_prior[il])
                    il += 1
            for _ in range(3):
                if ih < len(X_h):
                    comb_X.append(X_h[ih])
                    comb_Y.append(Y_h[ih])
                    comb_U.append(U_h[ih])
                    comb_prior.append(w_h_prior[ih])
                    ih += 1

        self.crystal_X = np.asarray(comb_X, dtype=np.float64)
        self.crystal_Y = np.asarray(comb_Y, dtype=np.float64).squeeze()
        self.crystal_U = np.asarray(comb_U, dtype=np.float64)
        prior_vec = np.asarray(comb_prior, dtype=np.float64)
        M_curr = len(self.crystal_U)

        # Спектральная квадратура FISTA вокруг несмещенного распределения Радона-Никодима
        Psi_hat = self.Psi_sum / self.n_seen
        G_U = self.crystal_U @ self.crystal_U.T
        K = (G_U ** 2) + 0.15 * np.eye(M_curr)
        b = np.sum((self.crystal_U @ Psi_hat) * self.crystal_U, axis=1) + 0.15 * prior_vec

        L_lip = float(linalg.norm(K, ord=2)) + 1e-6
        gamma = prior_vec.copy()
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

        # Синтез несмещенной плотности Вороного и спектральной квадратуры (со средним 1.0)
        blended_w = 0.65 * (prior_vec * M_curr) + 0.35 * (gamma_prob * M_curr)
        self.gamma_weights = np.clip(blended_w, 0.25, 2.80)
        self.gamma_weights /= np.mean(self.gamma_weights)
        return rel_error

    def save_hologram(self, filepath: str = "superconduct_v35.hologram.npz", rank_passport: int = 32) -> float:
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


# --- 2. ЗАГРУЗКА ДАТАСЕТА И ОЧИСТКА ПАМЯТИ ПРОТИВ OOM ---
gc.collect()
if "X_raw" not in globals():
    try:
        ds = fetch_openml(name="superconduct", version=1, as_frame=False, parser="auto")
        X_raw = np.asarray(ds.data, dtype=np.float64)
        y_raw = np.asarray(ds.target, dtype=np.float64)
        crit_threshold = 77.0
        unit_name = "Кельвин (K)"
    except Exception:
        ds = fetch_california_housing()
        X_raw = np.asarray(ds.data, dtype=np.float64)
        y_raw = np.asarray(ds.target, dtype=np.float64)
        crit_threshold = 4.2
        unit_name = "x$100k"

X_train_stream, X_test_all, y_train_stream, y_test_all = train_test_split(
    X_raw, y_raw, test_size=0.20, random_state=42
)
is_test_crit = y_test_all >= crit_threshold
X_test_norm, y_test_norm = X_test_all[~is_test_crit], y_test_all[~is_test_crit]
X_test_crit, y_test_crit = X_test_all[is_test_crit], y_test_all[is_test_crit]


# --- 3. ПОТОКОВАЯ ПЕРЕГОНКА SHD v3.5 ---
M_CRYSTAL = 300
BATCH_SIZE = 2500

tracemalloc.start()
t0 = time.perf_counter()

distiller = SpectralHologramDistillerV35(
    crystal_size=M_CRYSTAL, high_band_ratio=0.30, sketch_dim=256, tail_quantile=85.0
)

for start_i in range(0, len(X_train_stream), BATCH_SIZE):
    end_i = min(start_i + BATCH_SIZE, len(X_train_stream))
    distiller.partial_fit(X_train_stream[start_i:end_i], y_train_stream[start_i:end_i])

rel_spec_error = distiller.finalize_crystal()
elapsed = time.perf_counter() - t0
_, peak_mem = tracemalloc.get_traced_memory()
tracemalloc.stop()

holo_kb = distiller.save_hologram("superconduct_v35.hologram.npz")
raw_kb = (X_train_stream.nbytes + y_train_stream.nbytes) / 1024.0

print("=" * 98)
print(f"ТЕЛЕМЕТРИЯ SHD v3.5 С КОРРЕКЦИЕЙ РАДОНА-НИКОДИМА ({M_CRYSTAL} точек = 1.76%)")
print("=" * 98)
print(f"Время на CPU: {elapsed:.2f} сек | Пик RAM: {peak_mem / (1024**2):.2f} МБ | Ошибка спектра: {rel_spec_error*100:.2f}%")
print(f"Размер файла: {holo_kb:.1f} КБ вместо {raw_kb:.0f} КБ (Сжатие в {raw_kb/holo_kb:.0f} раз!)")
print(f"Захвачено ВТСП (Tc >= 77K): {np.sum(distiller.crystal_Y >= crit_threshold)} шт. | Диапазон весов gamma*: [{np.min(distiller.gamma_weights):.2f}, {np.max(distiller.gamma_weights):.2f}]\n")


# --- 4. СРАВНЕНИЕ МОДЕЛЕЙ (БЕЗ OOM-ПАДЕНИЯ COLAB) ---
rng_b = np.random.default_rng(42)
rand_idx = rng_b.choice(len(X_train_stream), size=M_CRYSTAL, replace=False)
X_rand, y_rand = X_train_stream[rand_idx], y_train_stream[rand_idx]

std_scaler = StandardScaler().fit(X_train_stream)
Z_train_all = std_scaler.transform(X_train_stream)
kmeans = MiniBatchKMeans(n_clusters=M_CRYSTAL, batch_size=2048, random_state=42, n_init=3).fit(Z_train_all)
km_counts = np.bincount(kmeans.labels_, minlength=M_CRYSTAL).astype(np.float64) + 1.0
km_weights = km_counts / np.mean(km_counts)
d_km = cdist(kmeans.cluster_centers_, Z_train_all, metric="sqeuclidean")
km_idx = np.argmin(d_km, axis=1)
X_km, y_km = X_train_stream[km_idx], y_train_stream[km_idx]

Z_test_all = std_scaler.transform(X_test_all)
Z_test_crit = std_scaler.transform(X_test_crit)
rbf_gamma = 1.0 / (X_raw.shape[1] * 1.1)

print("=" * 98)
print("БЕНЧМАРК 1: ЯДЕРНАЯ РЕГРЕССИЯ (RBF KernelRidge) НА ФИЗИКЕ СВЕРХПРОВОДНИКОВ")
print("=" * 98)
print(f"{'Обучающая выборка':<32} | {'RMSE (Все)':<12} | {'RMSE (Пик Tc>=77K)':<20} | {'R² (Общий)':<12} | {'Время обучения'}")
print("-" * 98)

# Для KernelRidge ограничиваем Oracle 9000 строками, чтобы матрица 9000x9000 не роняла RAM Colab
for name, (X_tr, y_tr, w_tr) in [
    ("Oracle (Полный датасет)", (X_train_stream[:9000], y_train_stream[:9000], None)),
    ("Random Coreset (1.76%)", (X_rand, y_rand, None)),
    ("K-Means Coreset (1.76%)", (X_km, y_km, km_weights)),
    ("Spectral Hologram v3.5 (1.76%)", (distiller.crystal_X, distiller.crystal_Y, distiller.gamma_weights)),
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
    ("Spectral Hologram v3.5 (1.76%)", (distiller.crystal_X, distiller.crystal_Y, distiller.gamma_weights)),
]:
    t_s = time.perf_counter()
    gbm = HistGradientBoostingRegressor(max_iter=120, max_leaf_nodes=20, min_samples_leaf=5, random_state=42)
    gbm.fit(X_tr, y_tr, sample_weight=w_tr)
    fit_ms = (time.perf_counter() - t_s) * 1000.0

    pred_all = gbm.predict(X_test_all)
    pred_crit = gbm.predict(X_test_crit)
    print(f"{name:<32} | {np.sqrt(mean_squared_error(y_test_all, pred_all)):<12.2f} | {np.sqrt(mean_squared_error(y_test_crit, pred_crit)):<20.2f} | {r2_score(y_test_all, pred_all):<12.4f} | {fit_ms:.1f} мс")