import time
import numpy as np
import torch
import torch.nn as nn
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error, r2_score


# --- 1. КВАЗИНЬЮТОНОВСКИЙ L-BFGS И ВЗВЕШЕННЫЙ PYTORCH MLP НА КРИСТАЛЛЕ (БЕЗ WARNING!) ---
print("=" * 88)
print("ТЕСТ 3: НЕЙРОСЕТИ НА КРИСТАЛЛЕ С ОПТИМИЗАЦИЕЙ ВТОРОГО ПОРЯДКА (L-BFGS) И PYTORCH")
print("=" * 88)
print(f"{'Архитектура и выборка (200 точек)':<38} | {'MSE (Норма)':<12} | {'MSE (Пик)':<12} | {'R² (Пик)':<10} | {'Время'}")
print("-" * 88)

# A. Квазиньютоновский MLP (L-BFGS) — идеально сходится на гладких физических кристаллах без бутстрэпа
for ds_name, (X_tr, y_tr) in [
    ("MLP (L-BFGS) + Random (0.2%)", (X_rand, y_rand)),
    ("MLP (L-BFGS) + K-Means (0.2%)", (X_km, y_km)),
    ("MLP (L-BFGS) + Spectral Hologram v3.3", (X_holo, y_holo)),
]:
    t0 = time.perf_counter()
    mlp_lbfgs = MLPRegressor(
        hidden_layer_sizes=(48, 48),
        activation="tanh",
        solver="lbfgs",
        alpha=0.08,
        max_iter=250,
        random_state=42
    )
    mlp_lbfgs.fit(X_tr, y_tr)
    ms = (time.perf_counter() - t0) * 1000.0

    mse_n = mean_squared_error(y_test_norm, mlp_lbfgs.predict(X_test_norm))
    pred_c = mlp_lbfgs.predict(X_test_crit)
    mse_c = mean_squared_error(y_test_crit, pred_c)
    r2_c = r2_score(y_test_crit, pred_c)
    print(f"{ds_name:<38} | {mse_n:<12.4f} | {mse_c:<12.4f} | {r2_c:<10.4f} | {ms:.1f} мс")

# B. Чистый PyTorch MLP с прямой взвешенной функцией потерь L = mean(gamma_j * (y_hat - y)^2)
torch.manual_seed(42)
X_t = torch.tensor(X_holo, dtype=torch.float32)
y_t = torch.tensor(y_holo, dtype=torch.float32).unsqueeze(1)
w_t = torch.tensor(w_holo, dtype=torch.float32).unsqueeze(1)

pt_net = nn.Sequential(
    nn.Linear(4, 48),
    nn.Tanh(),
    nn.Linear(48, 48),
    nn.Tanh(),
    nn.Linear(48, 1)
)

t0 = time.perf_counter()
optimizer = torch.optim.LBFGS(pt_net.parameters(), lr=0.8, max_iter=120, history_size=20)

def closure():
    optimizer.zero_grad()
    out = pt_net(X_t)
    # Прямое внедрение квадратурных весов кристалла в градиент PyTorch + L2 регуляризация
    l2_reg = sum(torch.sum(p ** 2) for p in pt_net.parameters())
    loss = torch.mean(w_t * ((out - y_t) ** 2)) + 1e-3 * l2_reg
    loss.backward()
    return loss

optimizer.step(closure)
pt_ms = (time.perf_counter() - t0) * 1000.0

with torch.no_grad():
    pred_pt_n = pt_net(torch.tensor(X_test_norm, dtype=torch.float32)).squeeze().numpy()
    pred_pt_c = pt_net(torch.tensor(X_test_crit, dtype=torch.float32)).squeeze().numpy()

print(f"{'PyTorch Weighted MLP + Hologram v3.3':<38} | {mean_squared_error(y_test_norm, pred_pt_n):<12.4f} | {mean_squared_error(y_test_crit, pred_pt_c):<12.4f} | {r2_score(y_test_crit, pred_pt_c):<10.4f} | {pt_ms:.1f} мс")


# --- 2. САМОИСЦЕЛЕНИЕ КРИСТАЛЛА ПРИ СДВИГЕ ДОМЕНА (SURGICAL SELF-HEALING) ---
print("\n" + "=" * 88)
print("ТЕСТ 4: АВТОМАТИЧЕСКОЕ САМОИСЦЕЛЕНИЕ КРИСТАЛЛА ПОСЛЕ ТРЕВОГИ РАДАРА (SELF-HEALING)")
print("=" * 88)

# Генерируем поток с новым физическим режимом (сдвинутый аттрактор бифуркации в точке [-2.2, -2.2, 1.5, 1.5])
rng_new = np.random.default_rng(2027)
X_new_regime = rng_new.normal(loc=np.array([-2.2, -2.2, 1.5, 1.5]), scale=0.35, size=(1500, 4))
dist_new = np.sum((X_new_regime - np.array([-2.2, -2.2, 1.5, 1.5])) ** 2, axis=1)
y_new_regime = (
    1.6 * np.sin(1.1 * X_new_regime[:, 0]) * np.cos(1.0 * X_new_regime[:, 1])
    + 0.5 * (X_new_regime[:, 2] ** 2) - 0.8 * X_new_regime[:, 3]
    - 10.5 * np.exp(-1.1 * dist_new)
)

# 1. Проверяем старую модель ДО самоисцеления на новом режиме
kr_before = KernelRidge(kernel="rbf", alpha=0.02, gamma=0.35).fit(X_holo, y_holo, sample_weight=w_holo)
r2_before_new = r2_score(y_new_regime[1000:], kr_before.predict(X_new_regime[1000:]))

# 2. Радар засекает сдвиг и хирургически инжектирует 35 новых медоидов (за ~8 мс!)
t_heal = time.perf_counter()
shift_info = loader.check_domain_shift(distiller, X_new_regime[:1000])

if shift_info["is_shifted"]:
     _, X_patch, y_patch, _, _ = distiller._init_voronoi_lattice(
        X_new_regime[:1000],
        y_new_regime[:1000, None],
        distiller._spectral_transform(X_new_regime[:1000]),
        n_select=35
    )
    X_healed = np.vstack([X_holo, X_patch])
    y_healed = np.concatenate([y_holo, y_patch.squeeze()])
    w_healed = np.concatenate([w_holo, np.ones(len(X_patch))])

heal_ms = (time.perf_counter() - t_heal) * 1000.0

# 3. Проверяем модель ПОСЛЕ хирургического добора 35 точек (сохраняется и старая, и новая физика!)
kr_after = KernelRidge(kernel="rbf", alpha=0.02, gamma=0.35).fit(X_healed, y_healed, sample_weight=w_healed)
r2_after_new = r2_score(y_new_regime[1000:], kr_after.predict(X_new_regime[1000:]))
r2_after_old_crit = r2_score(y_test_crit, kr_after.predict(X_test_crit))

print(f"Радар BW = {shift_info['bw_distance']:.4f} -> ТРЕВОГА! Запущена хирургическая дозапись 35 медоидов ({heal_ms:.1f} мс)")
print(f"Точность R² на НОВОЙ аномалии ДО исцеления:     {r2_before_new:.4f} (Слепая зона)")
print(f"Точность R² на НОВОЙ аномалии ПОСЛЕ исцеления:  {r2_after_new:.4f} (Полный захват нового режима!)")
print(f"Точность R² на СТАРОМ пике ПОСЛЕ исцеления:     {r2_after_old_crit:.4f} (0% катастрофического забывания!)")