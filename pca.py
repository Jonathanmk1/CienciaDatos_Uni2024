import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
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

# Aplicar PCA
pca = PCA(n_components=1)  # Reducimos a 1 componente principal
X_pca = pca.fit_transform(X_scaled)

# Visualizar la variancia explicada
print(f'Varianza explicada por el componente principal: {pca.explained_variance_ratio_}')

# Graficar los resultados
plt.scatter(X_pca, np.zeros_like(X_pca), c='blue')
plt.title('PCA - Población Hombres y Mujeres')
plt.xlabel('Componente Principal 1')
plt.ylabel('Valores Escalados')
plt.show()
