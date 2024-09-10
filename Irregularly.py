from scipy.spatial import Delaunay
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
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

def grafico_contornos_rellenos_irregulares(data_cleaned):
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
    ax.plot(x, y, 'o', markersize=8, color='grey')

    # Graficar los contornos rellenos
    contourf = ax.tricontourf(x, y, z, levels=np.linspace(z.min(), z.max(), 7), cmap='viridis')

    # Añadir una barra de colores
    cbar = plt.colorbar(contourf, ax=ax)
    cbar.set_label('Población Total')

    # Configurar límites y etiquetas
    ax.set_title("Contornos Rellenos de Población Total por Estado (INEGI)")
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_xlim(0, 25)
    ax.set_ylim(0, 25)

    plt.tight_layout()
    plt.show()

def grafico_tripcolor(data_cleaned):
    # Seleccionar algunos estados para el ejemplo (25 estados)
    estados = data_cleaned['Nombre Estado'][:25]
    poblacion_total = data_cleaned['Total'][:25]

    # Generar datos para una malla irregular
    np.random.seed(1)
    x = np.random.uniform(0, 25, 25)  # Posiciones X irregulares
    y = np.random.uniform(0, 25, 25)  # Posiciones Y irregulares
    z = poblacion_total.values

    # Triangulación de Delaunay
    triang = Delaunay(np.column_stack((x, y)))

    # Crear el gráfico
    fig, ax = plt.subplots()

    # Graficar los puntos de datos
    ax.plot(x, y, 'o', markersize=8, color='grey')

    # Graficar el tripcolor
    pc = ax.tripcolor(x, y, triang.simplices, z, shading='flat', cmap='viridis')

    # Añadir una barra de colores
    cbar = plt.colorbar(pc, ax=ax)
    cbar.set_label('Población Total')

    # Configurar límites y etiquetas
    ax.set_title("Coloración de Población Total por Estado (INEGI) usando Tripcolor")
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_xlim(0, 25)
    ax.set_ylim(0, 25)

    plt.tight_layout()
    plt.show()

def grafico_triplot(data_cleaned):
    # Seleccionar algunos estados para el ejemplo (25 estados)
    estados = data_cleaned['Nombre Estado'][:25]
    poblacion_total = data_cleaned['Total'][:25]

    # Generar datos para una malla irregular
    np.random.seed(1)
    x = np.random.uniform(0, 25, 25)  # Posiciones X irregulares
    y = np.random.uniform(0, 25, 25)  # Posiciones Y irregulares

    # Triangulación de Delaunay
    triang = Delaunay(np.column_stack((x, y)))

    # Crear el gráfico
    fig, ax = plt.subplots()

    # Graficar la malla triangular
    ax.triplot(x, y, triang.simplices, 'go-', markersize=5, linewidth=1)

    # Añadir etiquetas para los puntos
    for i in range(len(x)):
        ax.text(x[i], y[i], f'{i}', fontsize=8, ha='right', color='black')

    # Configurar límites y etiquetas
    ax.set_title("Malla Triangular con Triplot (INEGI)")
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_xlim(0, 25)
    ax.set_ylim(0, 25)

    plt.tight_layout()
    plt.show()

def grafico_barras_3d(data_cleaned):
    # Seleccionar algunos estados para el gráfico
    estados = data_cleaned['Nombre Estado'][:4]  # Solo 4 estados para simplicidad
    poblacion_total = data_cleaned['Total'][:4]

    # Crear coordenadas para el gráfico de barras 3D
    x = np.arange(len(estados))  # Coordenadas X
    y = np.zeros_like(x)  # Coordenadas Y
    z = np.zeros_like(x)  # Coordenadas Z
    dx = np.ones_like(x) * 0.5  # Ancho de las barras
    dy = np.ones_like(y) * 0.5  # Profundidad de las barras
    dz = poblacion_total  # Altura de las barras

    # Crear gráfico
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.bar3d(x, y, z, dx, dy, dz, shade=True)

    # Etiquetas del gráfico
    ax.set_xlabel('Estado')
    ax.set_ylabel('Categoría')
    ax.set_zlabel('Población Total')

    # Configurar etiquetas del eje X
    ax.set_xticks(x)
    ax.set_xticklabels(estados, rotation=45, ha='right')

    plt.show()

def menu():
    """Despliega el menú para seleccionar el tipo de gráfico."""
    print("Seleccione un gráfico:")
    print("1. tricontour(x, y, z)")
    print("2. tricontourf(x, y, z)")
    print("3. tripcolor(x, y, z)")
    print("4. triplot(x, y)")

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
        grafico_contornos_rellenos_irregulares(data_cleaned)
    elif opcion == '2':
        grafico_tripcolor(data_cleaned)
    elif opcion == '3':
        grafico_triplot(data_cleaned)
    elif opcion == '4':
        grafico_barras_3d(data_cleaned)
    else:
        print("Opción no válida. Por favor, selecciona un número del 1 al 4.")

if __name__ == "__main__":
    main()
