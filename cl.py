import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import dendrogram, linkage
from sklearn.cluster import AgglomerativeClustering

# Cargar y limpiar datos
file_path = 'INEGI_PoblacionTotal_Hombre_Mujeres_clasificados.xlsx'
df = pd.read_excel(file_path, sheet_name='hoja')
df_clean = df.dropna().copy()
df_clean['Hombres'] = df_clean['Unnamed: 3'].str.replace(',', '').astype(float)
df_clean['Mujeres'] = df_clean['Unnamed: 4'].str.replace(',', '').astype(float)

# Variables para el modelo (Hombres y Mujeres)
X = df_clean[['Hombres', 'Mujeres']].values

# Realizar el clustering jerárquico usando el método 'ward'
Z = linkage(X, method='ward')

# Graficar el dendrograma
plt.figure(figsize=(10, 7))
dendrogram(Z)
plt.title('Dendrograma de Clustering Jerárquico')
plt.xlabel('Estados')
plt.ylabel('Distancia')
plt.show()

# Aplicar Agglomerative Clustering para obtener los clusters
cluster_model = AgglomerativeClustering(n_clusters=3, affinity='euclidean', linkage='ward')
clusters = cluster_model.fit_predict(X)

# Visualizar los clusters en un gráfico de dispersión
plt.scatter(X[:, 0], X[:, 1], c=clusters, cmap='rainbow')
plt.xlabel('Población Hombres')
plt.ylabel('Población Mujeres')
plt.title('Clustering Jerárquico (n_clusters=3)')
plt.show()
