import os
import time
import tracemalloc
import numpy as np
import matplotlib.pyplot as plt
from scipy import linalg
from scipy.spatial.distance import cdist
from sklearn.cluster import MiniBatchKMeans
from sklearn.kernel_ridge import KernelRidge
from sklearn.metrics import mean_squared_error, r2_score


# --- 1. ПОТОКОВЫЙ ГЕНЕРАТОР 4D ФИЗИЧЕСКОГО РЕАКТОРА ---
def stream_reactor_dataset_4d(n_samples: int = 100_000, batch_size: int = 2500, seed: int = 101):
    """
    4-мерное фазовое пространство реактора (T, P, c1, c2):
    - 97.5% потока: гладкий термодинамический режим в кубе [-2, 2]^4
    - 2.5% потока: локализованный 4D-аттрактор критического фазового перехода
    """
    rng = np.random.default_rng(seed)
    crit_center = np.array([2.2, 2.2, 1.4, -1.4])
    n_delivered = 0

    while n_delivered < n_samples:
        B = min(batch_size, n_samples - n_delivered)
        X = rng.uniform(-2.0, 2.0, size=(B, 4))

        is_critical = rng.random(B) < 0.025
        n_crit = int(np.sum(is_critical))
        if n_crit > 0:
            X[is_critical] = rng.normal(loc=crit_center, scale=0.35, size=(n_crit, 4))

        smooth_law = (
            1.6 * np.sin(1.1 * X[:, 0]) * np.cos(1.0 * X[:, 1])
            + 0.5 * (X[:, 2] ** 2)
            - 0.8 * X[:, 3]
        )
        dist_sq = np.sum((X - crit_center) ** 2, axis=1)
        resonance_dome = 11.0 * np.exp(-1.1 * dist_sq) * np.cos(1.4 * X[:, 0] + 0.8 * X[:, 1])
        noise = rng.normal(0, 0.05, size=B)

        y = smooth_law + resonance_dome + noise
        n_delivered += B
        yield X, y, is_critical


# --- 2. ЯДРО SPECTRAL HOLOGRAM DISTILLER v3.2 (GOLDEN MASTER) ---
class SpectralHologramDistillerV32:
    """
    Spectral Hologram Distiller (SHD v3.2 — Golden Master)
    - Контур А (145 узлов): Полное покрытие нормального 4D-куба без краевых дыр
    - Контур Б (55 узлов): Захват эпицентра бифуркации (включая нулевые фазы волны cos(...) == 0)
    - Ранговое сжатие Спектрального Паспорта до ~15 КБ на диске
    """
    def __init__(
        self,
        crystal_size: int = 200,
        high_band_ratio: float = 0.275,  # 55 медоидов в эпицентре пика, 145 в основном кубе
        sketch_dim: int = 256,
        beta_low: float = 2e-3,
        surprise_threshold: float = 4.2,
        epicenter_radius_sq: float = 0.65,
        random_state: int = 42
    ):
        self.M = crystal_size
        self.M_high = int(round(crystal_size * high_band_ratio))
        self.M_low = crystal_size - self.M_high
        self.k = sketch_dim
        self.beta_low = beta_low
        self.surprise_thr = surprise_threshold
        self.epi_rad_sq = epicenter_radius_sq
        self.rng = np.random.default_rng(random_state)

        self.n_seen = 0
        self.Psi_sum = np.zeros((self.k, self.k), dtype=np.float64)
        self.Phi_sum = None
        self._proj_initialized = False

        # Контур А (Backbone — 145 узлов)
        self.low_centers_X = None
        self.low_X = None
        self.low_Y = None
        self.low_U = None
        self.low_counts = None

        # Контур Б (Phase-Peak — 55 узлов + центр масс эпицентра)
        self.high_pool_X, self.high_pool_Y, self.high_pool_U = [], [], []
        self.high_epicenter = None
        self.high_centers_X = None
        self.high_X = None
        self.high_Y = None
        self.high_U = None
        self.high_counts = None

        self.crystal_X = None
        self.crystal_Y = None
        self.crystal_U = None
        self.gamma_weights = None

    def _init_projector(self, X_batch: np.ndarray, y_dim: int):
        d = X_batch.shape[1]
        self.Phi_sum = np.zeros((self.k, y_dim), dtype=np.float64)
        self.X_mean = np.mean(X_batch, axis=0)
        self.X_std = np.std(X_batch, axis=0) + 1e-8

        k_half = self.k // 2
        omega_low = self.rng.normal(0, 0.65, size=(d, k_half))
        omega_high = self.rng.normal(0, 1.45, size=(d, self.k - k_half))
        self.Omega = np.hstack([omega_low, omega_high])
        self.phase = self.rng.uniform(0, 2 * np.pi, size=(1, self.k))
        self._proj_initialized = True

    def _spectral_transform(self, X_batch: np.ndarray) -> np.ndarray:
        X_norm = (X_batch - self.X_mean) / self.X_std
        U = np.sqrt(2.0 / self.k) * np.cos(X_norm @ self.Omega + self.phase)
        norms = np.linalg.norm(U, axis=1, keepdims=True) + 1e-8
        return U / norms

    def _init_voronoi_lattice(self, X_cand: np.ndarray, y_cand: np.ndarray, U_cand: np.ndarray, n_select: int):
        """
        Инициализация k-means++ с 4 быстрыми итерациями релаксации Ллойда
        и привязкой к ближайшим реальным физическим точкам (Medoids).
        """
        n_total = len(X_cand)
        first_idx = int(self.rng.integers(0, n_total))
        selected = [first_idx]
        min_sq_dists = np.sum((X_cand - X_cand[first_idx]) ** 2, axis=1)

        for _ in range(1, n_select):
            probs = min_sq_dists / (np.sum(min_sq_dists) + 1e-12)
            next_idx = int(self.rng.choice(n_total, p=probs))
            selected.append(next_idx)
            new_dists = np.sum((X_cand - X_cand[next_idx]) ** 2, axis=1)
            min_sq_dists = np.minimum(min_sq_dists, new_dists)

        centers = X_cand[selected].copy()

        # 4 шага релаксации сетки Вороного для идеального покрытия с первого батча
        for _ in range(4):
            d_mat = cdist(X_cand, centers, metric="sqeuclidean")
            labels = np.argmin(d_mat, axis=1)
            for c_i in range(n_select):
                mask = labels == c_i
                if np.any(mask):
                    centers[c_i] = np.mean(X_cand[mask], axis=0)

        # Привязываем центры к ближайшим реальным точкам (Medoids)
        d_final = cdist(centers, X_cand, metric="sqeuclidean")
        med_idx = np.argmin(d_final, axis=1)
        counts = np.bincount(labels, minlength=n_select).astype(np.float64) + 1.0

        return centers, X_cand[med_idx].copy(), y_cand[med_idx].copy(), U_cand[med_idx].copy(), counts

    @staticmethod
    def _update_voronoi_medoids(X_b, y_b, U_b, centers_X, med_X, med_Y, med_U, counts):
        if len(X_b) == 0:
            return
        dists = cdist(X_b, centers_X, metric="sqeuclidean")
        assign = np.argmin(dists, axis=1)

        for c_id in np.unique(assign):
            mask = assign == c_id
            n_new = int(np.sum(mask))
            counts[c_id] += n_new
            lr = min(0.25, n_new / counts[c_id])
            centers_X[c_id] = (1.0 - lr) * centers_X[c_id] + lr * np.mean(X_b[mask], axis=0)

            d_to_center = np.sum((X_b[mask] - centers_X[c_id]) ** 2, axis=1)
            best_local = int(np.argmin(d_to_center))
            cand_dist = float(d_to_center[best_local])
            curr_dist = float(np.sum((med_X[c_id] - centers_X[c_id]) ** 2))

            if cand_dist <= curr_dist:
                idx_global = np.where(mask)[0][best_local]
                med_X[c_id] = X_b[idx_global]
                med_Y[c_id] = y_b[idx_global]
                med_U[c_id] = U_b[idx_global]

    def partial_fit(self, X_batch: np.ndarray, y_batch: np.ndarray):
        if y_batch.ndim == 1:
            y_batch = y_batch[:, None]

        if not self._proj_initialized:
            self._init_projector(X_batch, y_batch.shape[1])

        B = X_batch.shape[0]
        U_B = self._spectral_transform(X_batch)

        self.Psi_sum += U_B.T @ U_B
        self.Phi_sum += U_B.T @ y_batch
        self.n_seen += B

        Psi_hat = self.Psi_sum / self.n_seen
        Phi_hat = self.Phi_sum / self.n_seen

        I_k = np.eye(self.k, dtype=np.float64)
        L_low = linalg.cho_factor(Psi_hat + self.beta_low * I_k, lower=True)
        W_pred = linalg.cho_solve(L_low, Phi_hat)

        residual = np.sum((y_batch - U_B @ W_pred) ** 2, axis=1)
        med_res = np.median(residual) + 1e-8
        res_ratio = residual / med_res

        # Чистый триггер высокой невязки
        is_spike_trigger = res_ratio > self.surprise_thr

        # --- КОНТУР А: Основной куб обучается на всех нормальных точках без вырезания углов ---
        X_norm = X_batch[~is_spike_trigger]
        y_norm = y_batch[~is_spike_trigger]
        U_norm = U_B[~is_spike_trigger]

        if self.low_X is None and len(X_norm) >= self.M_low:
            (
                self.low_centers_X,
                self.low_X,
                self.low_Y,
                self.low_U,
                self.low_counts,
            ) = self._init_voronoi_lattice(X_norm, y_norm, U_norm, self.M_low)
        elif self.low_X is not None:
            self._update_voronoi_medoids(
                X_norm, y_norm, U_norm,
                self.low_centers_X, self.low_X, self.low_Y, self.low_U, self.low_counts
            )

        # --- КОНТУР Б: Захват ядра бифуркации вокруг эпицентра (включая нули фазы) ---
        if self.high_epicenter is not None:
            d_epi = np.sum((X_batch - self.high_epicenter) ** 2, axis=1)
            in_high_core = is_spike_trigger | (d_epi <= self.epi_rad_sq)
            if np.any(is_spike_trigger):
                self.high_epicenter = 0.9 * self.high_epicenter + 0.1 * np.median(X_batch[is_spike_trigger], axis=0)
        else:
            in_high_core = is_spike_trigger

        X_sp = X_batch[in_high_core]
        y_sp = y_batch[in_high_core]
        U_sp = U_B[in_high_core]

        if self.high_X is None:
            if len(X_sp) > 0:
                self.high_pool_X.append(X_sp)
                self.high_pool_Y.append(y_sp)
                self.high_pool_U.append(U_sp)
                total_pooled = sum(len(arr) for arr in self.high_pool_X)
                if total_pooled >= self.M_high * 3:
                    Xp = np.vstack(self.high_pool_X)
                    Yp = np.vstack(self.high_pool_Y)
                    Up = np.vstack(self.high_pool_U)

                    self.high_epicenter = np.median(Xp, axis=0)
                    d_med = np.sum((Xp - self.high_epicenter) ** 2, axis=1)
                    inlier_mask = d_med <= np.percentile(d_med, 85)
                    if np.sum(inlier_mask) >= self.M_high:
                        Xp, Yp, Up = Xp[inlier_mask], Yp[inlier_mask], Up[inlier_mask]

                    (
                        self.high_centers_X,
                        self.high_X,
                        self.high_Y,
                        self.high_U,
                        self.high_counts,
                    ) = self._init_voronoi_lattice(Xp, Yp, Up, self.M_high)
                    self.high_pool_X, self.high_pool_Y, self.high_pool_U = [], [], []
        else:
            d_cand = np.sum((X_sp - self.high_epicenter) ** 2, axis=1)
            core_mask = d_cand <= (self.epi_rad_sq * 1.35)
            if np.any(core_mask):
                self._update_voronoi_medoids(
                    X_sp[core_mask], y_sp[core_mask], U_sp[core_mask],
                    self.high_centers_X, self.high_X, self.high_Y, self.high_U, self.high_counts
                )

    def finalize_crystal(self, fista_steps: int = 300):
        low_order = np.argsort(self.low_counts)[::-1]
        high_order = np.argsort(self.high_counts)[::-1]

        X_l, Y_l, U_l = self.low_X[low_order], self.low_Y[low_order], self.low_U[low_order]
        X_h, Y_h, U_h = self.high_X[high_order], self.high_Y[high_order], self.high_U[high_order]

        # Стратифицированное чередование (8 точек каркаса : 3 точки пика)
        comb_X, comb_Y, comb_U = [], [], []
        il, ih = 0, 0
        while il < len(X_l) or ih < len(X_h):
            for _ in range(8):
                if il < len(X_l):
                    comb_X.append(X_l[il])
                    comb_Y.append(Y_l[il])
                    comb_U.append(U_l[il])
                    il += 1
            for _ in range(3):
                if ih < len(X_h):
                    comb_X.append(X_h[ih])
                    comb_Y.append(Y_h[ih])
                    comb_U.append(U_h[ih])
                    ih += 1

        self.crystal_X = np.asarray(comb_X, dtype=np.float64)
        self.crystal_Y = np.asarray(comb_Y, dtype=np.float64).squeeze()
        self.crystal_U = np.asarray(comb_U, dtype=np.float64)
        M_curr = len(self.crystal_U)

        # Спектральная квадратура Адамара-Грама через FISTA
        Psi_hat = self.Psi_sum / self.n_seen
        G_U = self.crystal_U @ self.crystal_U.T
        K = (G_U ** 2) + 1e-4 * np.eye(M_curr)
        b = np.sum((self.crystal_U @ Psi_hat) * self.crystal_U, axis=1)

        L_lip = float(linalg.norm(K, ord=2)) + 1e-6
        gamma = np.ones(M_curr, dtype=np.float64) / M_curr
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

        raw_w = gamma_prob * M_curr
        self.gamma_weights = np.clip(0.92 + 0.08 * raw_w, 0.88, 1.15)
        self.gamma_weights /= np.mean(self.gamma_weights)

        return rel_error

    def save_hologram(self, filepath: str = "reactor_crystal_v32.hologram.npz", rank_passport: int = 32) -> float:
        """Упаковка голограммы со спектральным сжатием паспорта (Топ-32 собственных мод во float16)."""
        Psi_hat = self.Psi_sum / self.n_seen
        eigvals, eigvecs = linalg.eigh(Psi_hat)
        top_vals = eigvals[-rank_passport:].astype(np.float16)
        top_vecs = eigvecs[:, -rank_passport:].astype(np.float16)

        np.savez_compressed(
            filepath,
            crystal_X=self.crystal_X.astype(np.float32),
            crystal_Y=self.crystal_Y.astype(np.float32),
            gamma_weights=self.gamma_weights.astype(np.float32),
            passport_eigvals=top_vals,
            passport_eigvecs=top_vecs,
            X_mean=self.X_mean.astype(np.float32),
            X_std=self.X_std.astype(np.float32),
            n_seen=np.array([self.n_seen], dtype=np.int64)
        )
        return os.path.getsize(filepath) / 1024.0


# --- 3. ЗАПУСК ФИНАЛЬНОГО ИСПЫТАНИЯ SHD v3.2 ---
tracemalloc.start()
t0 = time.perf_counter()

distiller = SpectralHologramDistillerV32(crystal_size=200, high_band_ratio=0.275, sketch_dim=256)
X_full_list, y_full_list, crit_full_list = [], [], []

for X_b, y_b, crit_b in stream_reactor_dataset_4d(n_samples=100_000, batch_size=2500):
    distiller.partial_fit(X_b, y_b)
    X_full_list.append(X_b)
    y_full_list.append(y_b)
    crit_full_list.append(crit_b)

rel_spec_error = distiller.finalize_crystal()
elapsed = time.perf_counter() - t0
_, peak_mem = tracemalloc.get_traced_memory()
tracemalloc.stop()

file_kb = distiller.save_hologram("reactor_crystal_v32.hologram.npz")

X_full = np.vstack(X_full_list)
y_full = np.concatenate(y_full_list)
crit_full = np.concatenate(crit_full_list)
raw_size_kb = (X_full.nbytes + y_full.nbytes) / 1024.0

# Тестовая выборка
X_test_raw, y_test_raw, crit_test_raw = next(
    stream_reactor_dataset_4d(n_samples=40_000, batch_size=40_000, seed=2026)
)
idx_norm = np.where(~crit_test_raw)[0][:2000]
idx_crit = np.where(crit_test_raw)[0][:500]
X_test_norm, y_test_norm = X_test_raw[idx_norm], y_test_raw[idx_norm]
X_test_crit, y_test_crit = X_test_raw[idx_crit], y_test_raw[idx_crit]

M = 200
rng_b = np.random.default_rng(42)
rand_idx = rng_b.choice(len(X_full), size=M, replace=False)
X_rand, y_rand = X_full[rand_idx], y_full[rand_idx]

kmeans = MiniBatchKMeans(n_clusters=M, batch_size=2048, random_state=42, n_init=3).fit(X_full)
km_counts = np.bincount(kmeans.labels_, minlength=M).astype(np.float64) + 1.0
km_weights = km_counts / np.mean(km_counts)
km_indices = [np.argmin(np.sum((X_full[:15000] - c) ** 2, axis=1)) for c in kmeans.cluster_centers_]
X_km, y_km = X_full[km_indices], y_full[km_indices]

X_oracle, y_oracle = X_full[:15000], y_full[:15000]

models = {
    "Oracle (Полный датасет)": (X_oracle, y_oracle, None),
    "Random Coreset (0.2%)": (X_rand, y_rand, None),
    "K-Means Coreset (0.2%)": (X_km, y_km, km_weights),
    "Spectral Hologram v3.2 (0.2%)": (distiller.crystal_X, distiller.crystal_Y, distiller.gamma_weights)
}

results = {}
for name, (X_tr, y_tr, weights) in models.items():
    t_s = time.perf_counter()
    model = KernelRidge(kernel="rbf", alpha=0.02, gamma=0.35)
    model.fit(X_tr, y_tr, sample_weight=weights)
    fit_ms = (time.perf_counter() - t_s) * 1000

    pred_n = model.predict(X_test_norm)
    pred_c = model.predict(X_test_crit)

    results[name] = {
        "MSE_Normal": mean_squared_error(y_test_norm, pred_n),
        "MSE_Critical": mean_squared_error(y_test_crit, pred_c),
        "R2_Critical": r2_score(y_test_crit, pred_c),
        "Fit_Time_ms": fit_ms
    }

print(f"Потоковая перегонка 100k -> 200 точек: {elapsed:.2f} сек | Пик RAM: {peak_mem / (1024**2):.2f} МБ | Ошибка спектра: {rel_spec_error*100:.2f}%")
print(f"Артефакт сохранен: reactor_crystal_v32.hologram.npz ({file_kb:.1f} КБ вместо {raw_size_kb:.0f} КБ — сжатие файла в {raw_size_kb/file_kb:.0f} раз!)\n")
print(f"{'Метод обучения':<30} | {'MSE (Норма)':<12} | {'MSE (Фазовый пик)':<18} | {'R2 (Фазовый пик)':<16} | {'Время обучения'}")
print("-" * 100)
for name, m in results.items():
    print(f"{name:<30} | {m['MSE_Normal']:<12.4f} | {m['MSE_Critical']:<18.4f} | {m['R2_Critical']:<16.4f} | {m['Fit_Time_ms']:.1f} мс")

# --- 4. ВИЗУАЛИЗАЦИЯ ---
fig, axes = plt.subplots(1, 3, figsize=(18, 5.2))
plt.subplots_adjust(wspace=0.28)

ax0 = axes[0]
ax0.scatter(X_full[:4000, 0], X_full[:4000, 1], c="lightgray", s=6, alpha=0.4, label="Фоновый поток (97.5%)")
crit_pts = X_full[crit_full][:300]
ax0.scatter(crit_pts[:, 0], crit_pts[:, 1], c="#ff9f43", s=15, alpha=0.6, label="4D-аттрактор бифуркации (2.5%)")
ax0.scatter(X_rand[:, 0], X_rand[:, 1], c="#54a0ff", marker="x", s=35, label="Random (0.2%)")
sizes = 35 + 65 * distiller.gamma_weights
ax0.scatter(distiller.crystal_X[:, 0], distiller.crystal_X[:, 1], c="#ee5253", edgecolors="black", s=sizes, alpha=0.88, label="Spectral Hologram v3.2")
ax0.set_title("1. Покрытие пространства (SHD v3.2)", fontsize=11, fontweight="bold")
ax0.set_xlabel("Канал X0 (Температура)")
ax0.set_ylabel("Канал X1 (Давление)")
ax0.legend(loc="lower left", fontsize=8)
ax0.grid(alpha=0.25)

ax1 = axes[1]
names_short = ["Oracle\n(Полный)", "Spectral\nHologram v3.2", "K-Means\n(0.2%)", "Random\n(0.2%)"]
mse_vals = [
    results["Oracle (Полный датасет)"]["MSE_Critical"],
    results["Spectral Hologram v3.2 (0.2%)"]["MSE_Critical"],
    results["K-Means Coreset (0.2%)"]["MSE_Critical"],
    results["Random Coreset (0.2%)"]["MSE_Critical"]
]
colors = ["#10ac84", "#ee5253", "#feca57", "#54a0ff"]
bars = ax1.bar(names_short, mse_vals, color=colors, edgecolor="black", width=0.55)
for bar in bars:
    h = bar.get_height()
    ax1.text(bar.get_x() + bar.get_width() / 2, h + 0.06, f"{h:.2f}", ha="center", fontweight="bold", fontsize=9)
ax1.set_title("2. Ошибка MSE на фазовом пике (меньше = лучше)", fontsize=11, fontweight="bold")
ax1.set_ylabel("MSE на критических режимах")
ax1.set_ylim(0, max(mse_vals) * 1.18)
ax1.grid(axis="y", alpha=0.25)

ax2 = axes[2]
sub_sizes = [25, 50, 75, 100, 125, 150, 175, 200]
r2_shd_curve, r2_rand_curve = [], []
for k_sub in sub_sizes:
    m_shd = KernelRidge(kernel="rbf", alpha=0.02, gamma=0.35)
    m_shd.fit(distiller.crystal_X[:k_sub], distiller.crystal_Y[:k_sub], sample_weight=distiller.gamma_weights[:k_sub])
    r2_shd_curve.append(r2_score(y_test_crit, m_shd.predict(X_test_crit)))

    m_rnd = KernelRidge(kernel="rbf", alpha=0.02, gamma=0.35)
    m_rnd.fit(X_rand[:k_sub], y_rand[:k_sub])
    r2_rand_curve.append(r2_score(y_test_crit, m_rnd.predict(X_test_crit)))

ax2.plot(sub_sizes, r2_shd_curve, "o-", color="#ee5253", linewidth=2.5, label="Spectral Hologram v3.2")
ax2.plot(sub_sizes, r2_rand_curve, "s--", color="#54a0ff", linewidth=2.0, label="Random Sampling")
ax2.axhline(results["Oracle (Полный датасет)"]["R2_Critical"], color="#10ac84", linestyle=":", linewidth=2, label="Потолок Oracle")
ax2.set_title("3. Кривая Anytime-обучения (R² на пиках)", fontsize=11, fontweight="bold")
ax2.set_xlabel("Количество точек из кристалла")
ax2.set_ylabel("R² (выше = лучше)")
ax2.legend(loc="lower right", fontsize=8.5)
ax2.grid(alpha=0.25)

plt.tight_layout()
plt.show()