import numpy as np
import pandas as pd
from scipy import stats
import matplotlib.pyplot as plt


# ==========================================================
# 🔹 FUNCIÓN GENERAL PARA CALCULAR Y GRAFICAR KS
# ==========================================================
def ks_plot(df, score_col='score', target_col='target'):
    """
    Calcula y grafica el estadístico KS para un modelo de riesgo.

    Parámetros:
        df (DataFrame): datos con score y target
        score_col (str): nombre de la columna con el score del modelo
        target_col (str): nombre de la columna binaria (1 = evento, 0 = no evento)
    """

    # Validaciones básicas
    if score_col not in df.columns or target_col not in df.columns:
        raise ValueError("Las columnas especificadas no existen en el DataFrame.")

    # Separar las distribuciones
    data_good = df.loc[df[target_col] == 0, score_col]
    data_bad = df.loc[df[target_col] == 1, score_col]

    # Calcular KS
    ks_statistic, p_value = stats.ks_2samp(data_good, data_bad)
    print(f"KS Statistic: {ks_statistic:.3f}")
    print(f"P-valor: {p_value:.5f}")

    # Crear CDFs interpoladas
    data_good_sorted = np.sort(data_good)
    data_bad_sorted = np.sort(data_bad)

    cdf_good = np.arange(1, len(data_good_sorted) + 1) / len(data_good_sorted)
    cdf_bad = np.arange(1, len(data_bad_sorted) + 1) / len(data_bad_sorted)

    x_values = np.linspace(min(df[score_col]), max(df[score_col]), 1000)
    cdf_good_interp = np.interp(x_values, data_good_sorted, cdf_good)
    cdf_bad_interp = np.interp(x_values, data_bad_sorted, cdf_bad)

    # Calcular punto de máxima separación (KS)
    diff = np.abs(cdf_good_interp - cdf_bad_interp)
    ks_max_index = np.argmax(diff)
    ks_x = x_values[ks_max_index]
    ks_y1 = cdf_good_interp[ks_max_index]
    ks_y2 = cdf_bad_interp[ks_max_index]

    # Graficar
    plt.figure(figsize=(8, 6))
    plt.plot(x_values, cdf_good_interp, label='Buenos (No Default)', color='blue')
    plt.plot(x_values, cdf_bad_interp, label='Malos (Default)', color='orange')
    plt.vlines(ks_x, ks_y1, ks_y2, color='red', linestyle='--', label=f'KS Máximo = {ks_statistic:.3f}')

    plt.xlabel('Score del Modelo')
    plt.ylabel('Frecuencia acumulada')
    plt.title(f'Curvas CDF - KS = {ks_statistic:.3f}')
    plt.legend()
    plt.grid(True)
    plt.text(ks_x, (ks_y1 + ks_y2) / 2, f'KS = {ks_statistic:.3f}', color='red', fontsize=10, ha='left', va='center')
    plt.show()

    return ks_statistic


# ==========================================================
# 🔸 EJEMPLO DE USO CON TUS DATOS
# ==========================================================
# Supongamos que tienes un DataFrame real con columnas:
#   - "score" = probabilidad de default (de tu modelo)
#   - "target" = 1 si hubo default, 0 si no

# Ejemplo simulado (puedes reemplazarlo con tus datos reales)
np.random.seed(42)
df = pd.DataFrame({
    'score': np.random.rand(1000),  # scores del modelo
    'target': np.random.binomial(1, 0.3, 1000)  # 30% eventos (default)
})

# Llamar a la función
ks_value = ks_plot(df, score_col='score', target_col='target')
print(f"\n✅ KS del modelo: {ks_value:.3f}")