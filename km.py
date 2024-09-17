import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

# Cargar y limpiar datos
file_path = 'INEGI_PoblacionTotal_Hombre_Mujeres_clasificados.xlsx'
df = pd.read_excel(file_path, sheet_name='hoja')
df_clean = df.dropna().copy()
df_clean['Hombres'] = df_clean['Unnamed: 3'].str.replace(',', '').astype(float)
df_clean['Mujeres'] = df_clean['Unnamed: 4'].str.replace(',', '').astype(float)

# Variables para el modelo (usamos Hombres y Mujeres)
X = df_clean[['Hombres', 'Mujeres']].values

# Crear y entrenar el modelo KMeans
kmeans = KMeans(n_clusters=3, random_state=42)
kmeans.fit(X)

# Predecir los clusters
clusters = kmeans.predict(X)

# Visualizar los resultados
plt.scatter(X[:, 0], X[:, 1], c=clusters, cmap='viridis', label='Datos')
plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], s=300, c='red', label='Centroides')
plt.xlabel('Población Hombres')
plt.ylabel('Población Mujeres')
plt.legend()
plt.show()

print(f'Centroides de los clusters:\n{kmeans.cluster_centers_}')
