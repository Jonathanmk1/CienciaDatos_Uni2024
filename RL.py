import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix

# Generar datos simulados
np.random.seed(42)
X_sim = np.random.rand(100, 1) * 100000  # Valores de población total
y_sim = (X_sim > 50000).astype(int).ravel()  # Clase binaria: supera o no supera

# Dividir datos en entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X_sim, y_sim, test_size=0.3, random_state=42)

# Verificar clases en entrenamiento y prueba
print("Distribución de clases en el conjunto de entrenamiento:")
print(pd.Series(y_train).value_counts())

print("Distribución de clases en el conjunto de prueba:")
print(pd.Series(y_test).value_counts())

# Crear y entrenar el modelo
logistic_model = LogisticRegression()
logistic_model.fit(X_train, y_train)

# Predicciones y evaluación
y_pred = logistic_model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred, labels=[0, 1])

print(f'Accuracy: {accuracy}')
print(f'Confusion Matrix:\n{conf_matrix}')
