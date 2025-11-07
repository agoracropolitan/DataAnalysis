# ==========================================================
# 🔹 ANÁLISIS DE SENSIBILIDAD A VARIABLES CRÍTICAS
# ==========================================================

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Función logística
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

# Modelo base (ejemplo de modelo de PD)
def modelo_pd(desempleo, pib):
    """
    Modelo simplificado de probabilidad de default (PD):
    PD = sigmoid(0.5 + 1.2 * desempleo - 0.8 * pib)
    """
    return sigmoid(0.5 + 1.2 * desempleo - 0.8 * pib)

def sensibilidad_plot():
    """
    Analiza la sensibilidad del modelo frente a cambios en variables críticas:
    - Tasa de desempleo
    - Crecimiento del PIB
    """
    # Rango de desempleo y PIB
    desempleo_vals = np.linspace(0.02, 0.15, 100)  # 2% a 15%
    pib_vals = np.linspace(-0.05, 0.05, 100)       # -5% a +5%

    # PD base (escenario normal)
    pd_base = modelo_pd(0.07, 0.02)

    # Sensibilidad a desempleo
    pd_des = [modelo_pd(d, 0.02) for d in desempleo_vals]

    # Sensibilidad a PIB
    pd_pib = [modelo_pd(0.07, p) for p in pib_vals]

    # ---------------------------
    # Gráfico 1: desempleo
    # ---------------------------
    plt.figure(figsize=(8, 5))
    plt.plot(desempleo_vals * 100, pd_des, color='darkorange')
    plt.axvline(7, color='gray', linestyle='--', label='Escenario base (7%)')
    plt.xlabel('Tasa de Desempleo (%)')
    plt.ylabel('Probabilidad de Default (PD)')
    plt.title('Sensibilidad del modelo PD frente al Desempleo')
    plt.legend()
    plt.grid(True)
    plt.show()

    # ---------------------------
    # Gráfico 2: PIB
    # ---------------------------
    plt.figure(figsize=(8, 5))
    plt.plot(pib_vals * 100, pd_pib, color='green')
    plt.axvline(2, color='gray', linestyle='--', label='Escenario base (2%)')
    plt.xlabel('PIB (%)')
    plt.ylabel('Probabilidad de Default (PD)')
    plt.title('Sensibilidad del modelo PD frente al PIB')
    plt.legend()
    plt.grid(True)
    plt.show()

    # ---------------------------
    # Tabla de sensibilidad relativa
    # ---------------------------
    variaciones = [-0.1, 0, 0.1]  # ±10% de cambio
    resultados = []
    for delta in variaciones:
        pd_new = modelo_pd(0.07*(1+delta), 0.02)
        sensibilidad = (pd_new - pd_base) / pd_base / delta if delta != 0 else np.nan
        resultados.append(['Desempleo', delta, pd_new, sensibilidad])

    for delta in variaciones:
        pd_new = modelo_pd(0.07, 0.02*(1+delta))
        sensibilidad = (pd_new - pd_base) / pd_base / delta if delta != 0 else np.nan
        resultados.append(['PIB', delta, pd_new, sensibilidad])

    tabla = pd.DataFrame(resultados, columns=['Variable', 'Δ%', 'PD_nueva', 'Sensibilidad_relativa'])
    print("\n📊 Tabla de sensibilidad relativa:")
    print(tabla.round(4))


# ==========================================================
# 🔹 EJECUCIÓN DEL ANÁLISIS
# ==========================================================
if __name__ == "__main__":
    print("=== ANÁLISIS DE SENSIBILIDAD A VARIABLES CRÍTICAS ===")
    sensibilidad_plot()