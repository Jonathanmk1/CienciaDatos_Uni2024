import numpy as np
import pandas as pd
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import make_pipeline
import matplotlib.pyplot as plt

# Cargar y limpiar datos
file_path = 'INEGI_PoblacionTotal_Hombre_Mujeres_clasificados.xlsx'
df = pd.read_excel(file_path, sheet_name='hoja')
df_clean = df.dropna().copy()
df_clean['Total'] = df_clean['Unnamed: 2'].str.replace(',', '').astype(float)
df_clean['Hombres'] = df_clean['Unnamed: 3'].str.replace(',', '').astype(float)

# Variables para el modelo
X = df_clean['Hombres'].values.reshape(-1, 1)  # Feature: Hombres
y = df_clean['Total'].values  # Target: Total

# Crear y ajustar el modelo polinomial
degree = 3
polynomial_model = make_pipeline(PolynomialFeatures(degree), LinearRegression())
polynomial_model.fit(X, y)

# Predicción
X_new = np.linspace(X.min(), X.max(), 100).reshape(-1, 1)
y_pred = polynomial_model.predict(X_new)

# Graficar
plt.scatter(X, y, color='blue', label='Datos reales')
plt.plot(X_new, y_pred, color='green', label=f'Regresión polinomial (grado {degree})')
plt.xlabel('Población Hombres')
plt.ylabel('Población Total')
plt.legend()
plt.show()
