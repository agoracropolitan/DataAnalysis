# ==========================================================
# 🔹 ANÁLISIS DE SENSIBILIDAD + STRESS TESTING + EL/EC
# ==========================================================

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ----------------------------------------------------------
# 1️⃣ Modelo de Probabilidad de Default (PD)
# ----------------------------------------------------------
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def modelo_pd(desempleo, pib):
    """
    Modelo simplificado de probabilidad de default (PD):
    PD = sigmoid(0.5 + 1.2 * desempleo - 0.8 * pib)
    """
    return sigmoid(0.5 + 1.2 * desempleo - 0.8 * pib)

# ----------------------------------------------------------
# 2️⃣ Análisis de Sensibilidad
# ----------------------------------------------------------
def sensibilidad_plot():
    """
    Analiza cómo cambia la PD frente a variaciones del desempleo y del PIB.
    """
    desempleo_vals = np.linspace(0.02, 0.15, 100)
    pib_vals = np.linspace(-0.05, 0.05, 100)
    pd_base = modelo_pd(0.07, 0.02)

    pd_des = [modelo_pd(d, 0.02) for d in desempleo_vals]
    pd_pib = [modelo_pd(0.07, p) for p in pib_vals]

    # Gráfico: sensibilidad al desempleo
    plt.figure(figsize=(8, 5))
    plt.plot(desempleo_vals * 100, pd_des, color='darkorange')
    plt.axvline(7, color='gray', linestyle='--', label='Escenario base (7%)')
    plt.xlabel('Tasa de Desempleo (%)')
    plt.ylabel('Probabilidad de Default (PD)')
    plt.title('Sensibilidad del modelo PD frente al Desempleo')
    plt.legend()
    plt.grid(True)
    plt.show()

    # Gráfico: sensibilidad al PIB
    plt.figure(figsize=(8, 5))
    plt.plot(pib_vals * 100, pd_pib, color='green')
    plt.axvline(2, color='gray', linestyle='--', label='Escenario base (2%)')
    plt.xlabel('PIB (%)')
    plt.ylabel('Probabilidad de Default (PD)')
    plt.title('Sensibilidad del modelo PD frente al PIB')
    plt.legend()
    plt.grid(True)
    plt.show()

    # Tabla resumen de sensibilidad
    variaciones = [-0.1, 0, 0.1]
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

# ----------------------------------------------------------
# 3️⃣ Stress Testing con EL y EC
# ----------------------------------------------------------
def stress_testing():
    """
    Evalúa el comportamiento del modelo bajo escenarios macroeconómicos,
    incluyendo Expected Loss (EL) y Capital Económico (EC).
    """
    # Escenarios hipotéticos
    escenarios = {
        'Base':      {'desempleo': 0.07, 'pib': 0.02},
        'Recesión':  {'desempleo': 0.12, 'pib': -0.03},
        'Crisis':    {'desempleo': 0.15, 'pib': -0.06},
        'Optimista': {'desempleo': 0.05, 'pib': 0.04}
    }

    # Parámetros financieros del portafolio
    EAD = 1_000_000_000   # Exposure at Default = 1.000 millones
    LGD = 0.45             # Loss Given Default = 45%
    EC_factor = 3.0        # Factor multiplicador para Capital Económico (≈ 99.9% quantil)

    resultados = []
    for nombre, vars in escenarios.items():
        desempleo = vars['desempleo']
        pib = vars['pib']
        pd_esc = modelo_pd(desempleo, pib)

        # Expected Loss y Capital Económico
        EL = pd_esc * LGD * EAD
        EC = (pd_esc * (1 + EC_factor * np.sqrt(pd_esc * (1 - pd_esc))) * LGD * EAD) - EL

        resultados.append([
            nombre,
            desempleo,
            pib,
            pd_esc,
            EL,
            EC
        ])

    df_stress = pd.DataFrame(resultados, columns=['Escenario', 'Desempleo', 'PIB', 'PD', 'Expected_Loss', 'Capital_Económico'])
    print("\n💥 Resultados del Stress Testing (con EL y EC):")
    print(df_stress.round(4))

    # Gráfico: PD por escenario
    plt.figure(figsize=(8, 5))
    plt.bar(df_stress['Escenario'], df_stress['PD'], color=['blue', 'orange', 'red', 'green'])
    plt.ylabel('Probabilidad de Default (PD)')
    plt.title('Stress Testing de PD bajo escenarios macroeconómicos')
    plt.grid(axis='y')
    plt.show()

    # Gráfico: Expected Loss y Capital Económico
    plt.figure(figsize=(8, 5))
    plt.bar(df_stress['Escenario'], df_stress['Expected_Loss'] / 1e6, label='Expected Loss (MM$)', alpha=0.7)
    plt.bar(df_stress['Escenario'], df_stress['Capital_Económico'] / 1e6, label='Capital Económico (MM$)', alpha=0.7)
    plt.ylabel('Millones de dólares')
    plt.title('Expected Loss vs Capital Económico por escenario')
    plt.legend()
    plt.grid(axis='y')
    plt.show()

    return df_stress

# ----------------------------------------------------------
# 4️⃣ Ejecución completa
# ----------------------------------------------------------
if __name__ == "__main__":
    print("=== ANÁLISIS DE SENSIBILIDAD ===")
    sensibilidad_plot()

    print("\n=== STRESS TESTING CON EXPECTED LOSS Y CAPITAL ECONÓMICO ===")
    df = stress_testing()
