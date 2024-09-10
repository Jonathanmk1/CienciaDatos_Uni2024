import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def cargar_datos(file_path):
    """Carga el archivo Excel y devuelve el DataFrame."""
    data = pd.read_excel(file_path, engine='openpyxl')
    return data


def limpiar_datos(data):
    """Limpia el DataFrame eliminando filas innecesarias y ajustando los datos."""
    data_cleaned = data.drop([0, 1])  # Eliminar las dos primeras filas
    data_cleaned.columns = ['Estado', 'Nombre Estado', 'Total', 'Hombres', 'Mujeres']
    data_cleaned['Total'] = pd.to_numeric(data_cleaned['Total'].str.replace(',', ''), errors='coerce')
    data_cleaned['Hombres'] = pd.to_numeric(data_cleaned['Hombres'].str.replace(',', ''), errors='coerce')
    data_cleaned['Mujeres'] = pd.to_numeric(data_cleaned['Mujeres'].str.replace(',', ''), errors='coerce')
    return data_cleaned


def preparar_datos(data_cleaned):
    """Prepara los datos para graficar, seleccionando los primeros 25 estados."""
    estados = data_cleaned['Nombre Estado'][:25]
    poblacion_total = data_cleaned['Total'][:25]
    hombres = data_cleaned['Hombres'][:25]
    mujeres = data_cleaned['Mujeres'][:25]
    return estados, poblacion_total, hombres, mujeres

def grafico_mapa_calor(data_cleaned):
    # Seleccionar algunos estados para el ejemplo (25 estados)
    poblacion_total = data_cleaned['Total'][:25].values

    # Crear una matriz 5x5 (puedes ajustar el tamaño según tus datos)
    matrix_size = 5
    Z = np.reshape(poblacion_total, (matrix_size, matrix_size))

    # Crear el gráfico
    fig, ax = plt.subplots()

    # Mostrar la matriz como una imagen
    cax = ax.imshow(Z, origin='lower', cmap='viridis')

    # Añadir barra de color
    fig.colorbar(cax, ax=ax, label='Población Total')

    # Configurar título y etiquetas
    ax.set_title("Mapa de Calor de Población Total por Estado (INEGI)")
    ax.set_xlabel("Columna")
    ax.set_ylabel("Fila")

    plt.tight_layout()
    plt.show()
def grafico_mapa_calor_pcolormesh(data_cleaned):
    # Seleccionar algunos estados para el ejemplo (25 estados)
    poblacion_total = data_cleaned['Total'][:25].values

    # Crear una malla 5x5
    matrix_size = 5
    X = np.linspace(0, matrix_size - 1, matrix_size)
    Y = np.linspace(0, matrix_size - 1, matrix_size)
    X, Y = np.meshgrid(X, Y)

    # Ajustar los datos de población para que encajen en la malla
    Z = np.reshape(poblacion_total, (matrix_size, matrix_size))

    # Crear el gráfico
    fig, ax = plt.subplots()

    # Mostrar la malla como un mapa de colores
    cax = ax.pcolormesh(X, Y, Z, shading='auto', cmap='viridis')

    # Añadir barra de color
    fig.colorbar(cax, ax=ax, label='Población Total')

    # Configurar título y etiquetas
    ax.set_title("Mapa de Calor de Población Total por Estado (INEGI) con pcolormesh")
    ax.set_xlabel("X")
    ax.set_ylabel("Y")

    plt.tight_layout()
    plt.show()

def grafico_contornos_poblacion(data_cleaned):
    # Seleccionar algunos estados para el ejemplo (25 estados)
    poblacion_total = data_cleaned['Total'][:25].values

    # Crear una malla 5x5
    matrix_size = 5
    X = np.linspace(0, matrix_size - 1, matrix_size)
    Y = np.linspace(0, matrix_size - 1, matrix_size)
    X, Y = np.meshgrid(X, Y)

    # Ajustar los datos de población para que encajen en la malla
    Z = np.reshape(poblacion_total, (matrix_size, matrix_size))

    # Definir niveles para los contornos
    levels = np.linspace(np.min(Z), np.max(Z), 7)

    # Crear el gráfico
    fig, ax = plt.subplots()

    # Mostrar contornos en la malla
    contour = ax.contour(X, Y, Z, levels=levels, cmap='viridis')

    # Añadir barra de color
    fig.colorbar(contour, ax=ax, label='Población Total')

    # Configurar título y etiquetas
    ax.set_title("Contornos de Población Total por Estado (INEGI) con contour")
    ax.set_xlabel("X")
    ax.set_ylabel("Y")

    plt.tight_layout()
    plt.show()

def grafico_contornos_rellenos_poblacion(data_cleaned):
    # Seleccionar algunos estados para el ejemplo (25 estados)
    poblacion_total = data_cleaned['Total'][:25].values

    # Crear una malla 5x5
    matrix_size = 5
    X = np.linspace(0, matrix_size - 1, matrix_size)
    Y = np.linspace(0, matrix_size - 1, matrix_size)
    X, Y = np.meshgrid(X, Y)

    # Ajustar los datos de población para que encajen en la malla
    Z = np.reshape(poblacion_total, (matrix_size, matrix_size))

    # Definir niveles para los contornos
    levels = np.linspace(np.min(Z), np.max(Z), 7)

    # Crear el gráfico
    fig, ax = plt.subplots()

    # Mostrar contornos rellenos en la malla
    contourf = ax.contourf(X, Y, Z, levels=levels, cmap='viridis')

    # Añadir barra de color
    fig.colorbar(contourf, ax=ax, label='Población Total')

    # Configurar título y etiquetas
    ax.set_title("Contornos Rellenos de Población Total por Estado (INEGI) con contourf")
    ax.set_xlabel("X")
    ax.set_ylabel("Y")

    plt.tight_layout()
    plt.show()

def grafico_vectores_poblacion(data_cleaned):
    # Seleccionar algunos estados para el ejemplo (25 estados)
    poblacion_total = data_cleaned['Total'][:25].values

    # Crear una malla 5x5
    matrix_size = 5
    X, Y = np.meshgrid(np.arange(matrix_size), np.arange(matrix_size))

    # Convertir datos de población en una matriz 5x5
    Z = np.reshape(poblacion_total, (matrix_size, matrix_size))

    # Crear datos para vectores (U y V) con direcciones y magnitudes variadas
    np.random.seed(0)
    angles = np.random.uniform(0, 2 * np.pi, size=Z.shape)  # Ángulos aleatorios
    magnitudes = np.sqrt(Z)  # Magnitudes proporcionales a la raíz cuadrada de Z
    U = magnitudes * np.cos(angles)
    V = magnitudes * np.sin(angles)

    # Crear el gráfico
    fig, ax = plt.subplots()

    # Mostrar barbs en la malla
    ax.barbs(X, Y, U, V, barbcolor='C0', flagcolor='C0', length=7, linewidth=1.5)

    # Configurar límites y etiquetas
    ax.set_title("Vectores de Población Total por Estado (INEGI) con barbs")
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set(xlim=(0, matrix_size - 1), ylim=(0, matrix_size - 1))

    plt.tight_layout()
    plt.show()

def grafico_vectores_quiver(data_cleaned):
    # Seleccionar algunos estados para el ejemplo (25 estados)
    poblacion_total = data_cleaned['Total'][:25].values

    # Crear una malla 5x5
    matrix_size = 5
    X, Y = np.meshgrid(np.arange(matrix_size), np.arange(matrix_size))

    # Convertir datos de población en una matriz 5x5
    Z = np.reshape(poblacion_total, (matrix_size, matrix_size))

    # Crear datos para vectores (U y V) con magnitudes y direcciones variadas
    np.random.seed(0)
    angles = np.random.uniform(0, 2 * np.pi, size=Z.shape)  # Ángulos aleatorios
    magnitudes = Z / Z.max()  # Normalizar magnitudes para que se ajusten bien en el gráfico
    U = magnitudes * np.cos(angles)
    V = magnitudes * np.sin(angles)

    # Crear el gráfico
    fig, ax = plt.subplots()

    # Mostrar vectores con quiver
    ax.quiver(X, Y, U, V, color="C0", angles='xy', scale_units='xy', scale=1, width=.015)

    # Configurar límites y etiquetas
    ax.set_title("Campo de Vectores de Población Total por Estado (INEGI) con quiver")
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_xlim(-1, matrix_size)
    ax.set_ylim(-1, matrix_size)

    plt.tight_layout()
    plt.show()


# Cargar el archivo de Excel
file_path = 'INEGI_PoblacionTotal_Hombre_Mujeres_clasificados.xlsx'
data = pd.read_excel(file_path, engine='openpyxl')  # Especificar el motor 'openpyxl'

# Limpiar los datos
data_cleaned = data.drop([0, 1])  # Eliminar las dos primeras filas innecesarias
data_cleaned.columns = ['Estado', 'Nombre Estado', 'Total', 'Hombres', 'Mujeres']
data_cleaned['Total'] = pd.to_numeric(data_cleaned['Total'].str.replace(',', ''), errors='coerce')

# Seleccionar algunos estados para el ejemplo (25 estados)
poblacion_total = data_cleaned['Total'][:25].values

# Crear una malla 5x5
matrix_size = 5
X, Y = np.meshgrid(np.arange(matrix_size), np.arange(matrix_size))

# Convertir datos de población en una matriz 5x5
Z = np.reshape(poblacion_total, (matrix_size, matrix_size))

# Crear datos para el campo de flujo
# Utilizaremos gradientes para U y V para simular el flujo
U, V = np.gradient(Z)

# Crear el gráfico
fig, ax = plt.subplots()

# Mostrar el campo de flujo con streamplot
strm = ax.streamplot(X, Y, U, V, color=Z, linewidth=2, cmap='viridis', arrowstyle='-')

# Configurar límites y etiquetas
ax.set_title("Campo de Flujo de Población Total por Estado (INEGI) con streamplot")
ax.set_xlabel("X")
ax.set_ylabel("Y")
ax.set_xlim(-1, matrix_size)
ax.set_ylim(-1, matrix_size)

# Añadir una barra de colores
cbar = plt.colorbar(strm.lines, ax=ax)
cbar.set_label('Población Total')

plt.tight_layout()
plt.show()

def grafico_contornos_irregulares(data_cleaned):
    # Seleccionar algunos estados para el ejemplo (25 estados)
    estados = data_cleaned['Nombre Estado'][:25]
    poblacion_total = data_cleaned['Total'][:25]

    # Generar datos para una malla irregular
    np.random.seed(1)
    x = np.random.uniform(0, 25, 25)  # Posiciones X irregulares
    y = np.random.uniform(0, 25, 25)  # Posiciones Y irregulares
    z = poblacion_total.values

    # Crear el gráfico
    fig, ax = plt.subplots()

    # Graficar los puntos de datos
    ax.plot(x, y, 'o', markersize=8, color='lightgrey')

    # Graficar los contornos
    contour = ax.tricontour(x, y, z, levels=np.linspace(z.min(), z.max(), 7), cmap='viridis')

    # Añadir una barra de colores
    cbar = plt.colorbar(contour, ax=ax)
    cbar.set_label('Población Total')

    # Configurar límites y etiquetas
    ax.set_title("Contornos de Población Total por Estado (INEGI)")
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_xlim(0, 25)
    ax.set_ylim(0, 25)

    plt.tight_layout()
    plt.show()


def menu():
    """Despliega el menú para seleccionar el tipo de gráfico."""
    print("Seleccione un gráfico:")
    print("1. imshow(Z)")
    print("2. pcolormesh(X, Y, Z)")
    print("3. bcontour(X, Y, Z)")
    print("4. contourf(X, Y, Z)")
    print("5. barbs(X, Y, U, V)")
    print("6. quiver(X, Y, U, V)")
    print("7. streamplot(X, Y, U, V)")

    opcion = input("Ingresa el número de la opción: ")
    return opcion


def main():
    file_path = 'INEGI_PoblacionTotal_Hombre_Mujeres_clasificados.xlsx'

    # Cargar y limpiar los datos
    data = cargar_datos(file_path)
    data_cleaned = limpiar_datos(data)
    estados, poblacion_total, hombres, mujeres = preparar_datos(data_cleaned)

    # Menú para seleccionar el gráfico
    opcion = menu()

    if opcion == '1':
        grafico_mapa_calor(data_cleaned)
    elif opcion == '2':
        grafico_mapa_calor_pcolormesh(data_cleaned)
    elif opcion == '3':
        grafico_contornos_poblacion(data_cleaned)
    elif opcion == '4':
        grafico_contornos_rellenos_poblacion(data_cleaned)
    elif opcion == '5':
        grafico_vectores_poblacion(data_cleaned)
    elif opcion == '6':
        grafico_vectores_quiver(data_cleaned)
    elif opcion == '7':
        grafico_contornos_irregulares(data_cleaned)
    else:
        print("Opción no válida. Por favor, selecciona un número del 1 al 7.")

if __name__ == "__main__":
    main()
