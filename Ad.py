import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

# Cargar y limpiar datos
file_path = 'INEGI_PoblacionTotal_Hombre_Mujeres_clasificados.xlsx'
df = pd.read_excel(file_path, sheet_name='hoja')
df_clean = df.dropna().copy()
df_clean['Total'] = df_clean['Unnamed: 2'].str.replace(',', '').astype(float)
df_clean['Hombres'] = df_clean['Unnamed: 3'].str.replace(',', '').astype(float)

# Definir si la población total es mayor a un umbral
threshold = 1000000
df_clean['Supera_Umbral'] = (df_clean['Total'] > threshold).astype(int)

# Variables para el modelo
X = df_clean[['Hombres']].values  # Usando la población masculina como feature
y = df_clean['Supera_Umbral'].values  # Clase binaria (1: supera, 0: no supera)

# Dividir los datos en entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Crear y entrenar el modelo
tree_model = DecisionTreeClassifier(random_state=42)
tree_model.fit(X_train, y_train)

# Predicciones y evaluación
y_pred = tree_model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

# Visualizar el árbol de decisión
plt.figure(figsize=(12, 8))
plot_tree(tree_model, feature_names=['Hombres'], class_names=['No Supera', 'Supera'], filled=True)
plt.show()

print(f'Accuracy: {accuracy}')
