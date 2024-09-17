import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler

# Cargar y limpiar datos
file_path = 'INEGI_PoblacionTotal_Hombre_Mujeres_clasificados.xlsx'  # Cambia esta ruta
df = pd.read_excel(file_path, sheet_name='hoja')
df_clean = df.dropna().copy()
df_clean['Hombres'] = df_clean['Unnamed: 3'].str.replace(',', '').astype(float)
df_clean['Mujeres'] = df_clean['Unnamed: 4'].str.replace(',', '').astype(float)

# Seleccionar las columnas de interés
X = df_clean[['Hombres', 'Mujeres']].values

# Escalar los datos para normalizar
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Aplicar t-SNE
tsne = TSNE(n_components=2, random_state=42)
X_tsne = tsne.fit_transform(X_scaled)

# Graficar los resultados
plt.scatter(X_tsne[:, 0], X_tsne[:, 1], c='blue', label='Estados')
plt.title('t-SNE - Población Hombres y Mujeres')
plt.xlabel('Componente t-SNE 1')
plt.ylabel('Componente t-SNE 2')
plt.legend()
plt.show()
