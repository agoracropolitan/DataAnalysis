import numpy as np
import pandas as pd
import os

# Mostrar el directorio actual
print(os.getcwd())
# -----------------------------
# 1️⃣ Creamos datos de ejemplo
# -----------------------------
# Distribución original (entrenamiento)
#Parámetro	Significado	                        Ejemplo	    Explicación
#loc	    Media (μ) de la distribución	    loc=600	    Centro o valor esperado de los datos.
#scale	    Desviación estándar (σ)	            scale=50	Qué tan dispersos están los valores alrededor de la media.
#size	    Cantidad de valores a generar
#            (puede ser un número o una tupla)	size=10000  Numero total de muestras
#                                                o size=(100, 10)
train = np.random.normal(600, 50, 10000)  # media 600, std 50

# Distribución actual (producción) con leve desplazamiento
current = np.random.normal(620, 60, 10000)


# -----------------------------
# 2️⃣ Definimos función PSI
# -----------------------------
def calculate_psi(expected, actual, buckets=10):
    # Dividimos la variable en bins según los percentiles del conjunto base
    breakpoints = np.percentile(expected, np.arange(0, 100, 100 / buckets)) # np.arange returns ndarray
    print("breakpoints: ", breakpoints)
    print("percentiles: ", np.arange(0, 100, 100 / buckets))
    # percentile_value = np.percentile(data, 50) # Calculates the 50th percentile (median)
    # multiple_percentiles = np.percentile(data, [25, 50, 75]) # Calculates 25th, 50th, and 75th percentiles
    # Contamos las proporciones en cada bin
    expected_percents = np.histogram(expected, bins=breakpoints)[0] / len(expected)
    print("expected_percents: ", np.histogram(expected, bins=breakpoints))
    print("type expected_percents", type(expected_percents))
    actual_percents = np.histogram(actual, bins=breakpoints)[0] / len(actual)
    print("actual_percents: ", np.histogram(actual, bins=breakpoints))
    # Evitamos dividir por cero
    expected_percents = np.where(expected_percents == 0, 0.0001, expected_percents)
    actual_percents = np.where(actual_percents == 0, 0.0001, actual_percents)

    # Calculamos PSI
    psi_values = (expected_percents - actual_percents) * np.log(expected_percents / actual_percents)
    # np.log is a function within the NumPy library in Python used to calculate the natural logarithm (base-e logarithm) of a given input.
    psi = np.sum(psi_values)
    return psi


# -----------------------------
# 3️⃣ Ejecutamos el cálculo
# -----------------------------
psi_score = calculate_psi(train, current)
print(f"PSI = {psi_score:.3f}")
#2. La letra f antes de las comillas
# 👉 Indica que es una f-string (formatted string literal),
# una forma de insertar variables dentro de un texto directamente.
# 3. {psi_score:.3f}
#Esto es una expresión de formato numérico, que indica:
# psi_score → es la variable cuyo valor se insertará.
# :.3f → significa mostrar el número con 3 decimales, en formato de número flotante (float).
# Por ejemplo, si psi_score = 0.2538912,
# entonces print(f"PSI = {psi_score:.3f}") imprimirá:

# Interpretación típica:
if psi_score < 0.1:
    print("Sin cambio relevante")
elif psi_score < 0.25:
    print("Cambio moderado, revisar")
else:
    print("Cambio significativo, posible reentrenamiento")