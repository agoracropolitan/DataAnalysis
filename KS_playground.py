import numpy as np
import pandas as pd
from scipy import stats
import matplotlib.pyplot as plt

# -----------------------------
# 1️⃣ Simular datos (scores)
# -----------------------------
np.random.seed(42)
scores_buenos = np.random.normal(0.3, 0.1, 5000)  # clientes sin default
scores_malos = np.random.normal(0.6, 0.1, 5000)   # clientes con default

# Crear DataFrame
df = pd.DataFrame({
    'score': np.concatenate([scores_buenos, scores_malos]),
    'target': np.concatenate([np.zeros(5000), np.ones(5000)])  # 0=bueno, 1=malo
})

# -----------------------------
# 2️⃣ Calcular estadístico KS
# -----------------------------
data_good = df.loc[df['target'] == 0, 'score']
data_bad = df.loc[df['target'] == 1, 'score']

ks_statistic, p_value = stats.ks_2samp(data_good, data_bad)
print(f"KS Statistic: {ks_statistic:.3f}")
print(f"P-valor: {p_value:.5f}")

# -----------------------------
# 3️⃣ Graficar curvas acumuladas (CDF)
# -----------------------------
# Crear histogramas normalizados (CDF)
data_good_sorted = np.sort(data_good)
data_bad_sorted = np.sort(data_bad)

cdf_good = np.arange(1, len(data_good_sorted) + 1) / len(data_good_sorted)
cdf_bad = np.arange(1, len(data_bad_sorted) + 1) / len(data_bad_sorted)

# Alinear los ejes para comparar correctamente
x_values = np.linspace(min(df['score']), max(df['score']), 1000)
cdf_good_interp = np.interp(x_values, data_good_sorted, cdf_good)
cdf_bad_interp = np.interp(x_values, data_bad_sorted, cdf_bad)

# Calcular la diferencia absoluta y el punto de máxima separación
diff = np.abs(cdf_good_interp - cdf_bad_interp)
ks_max_index = np.argmax(diff)
ks_x = x_values[ks_max_index]
ks_y1 = cdf_good_interp[ks_max_index]
ks_y2 = cdf_bad_interp[ks_max_index]

# -----------------------------
# 4️⃣ Plot con línea de separación KS
# -----------------------------
plt.figure(figsize=(8, 6))
plt.plot(x_values, cdf_good_interp, label='Buenos (No Default)')
plt.plot(x_values, cdf_bad_interp, label='Malos (Default)')

# Línea vertical y conexión del punto KS
plt.vlines(ks_x, ks_y1, ks_y2, color='red', linestyle='--', label=f'KS Máximo = {ks_statistic:.3f}')

# Etiquetas y estilo
plt.xlabel('Score del Modelo')
plt.ylabel('Frecuencia acumulada')
plt.title(f'Curvas CDF - KS = {ks_statistic:.3f}')
plt.legend()
plt.grid(True)

# Anotación visual
plt.text(ks_x, (ks_y1 + ks_y2)/2, f'KS = {ks_statistic:.3f}', color='red', fontsize=10, ha='left', va='center')

plt.show()