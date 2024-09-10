import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


def cargar_datos(file_path):
    """Carga y limpia los datos del archivo Excel."""
    data = pd.read_excel(file_path, engine='openpyxl')
    data_cleaned = data.drop([0, 1])  # Eliminar las dos primeras filas innecesarias
    data_cleaned.columns = ['Estado', 'Nombre Estado', 'Total', 'Hombres', 'Mujeres']
    data_cleaned['Total'] = pd.to_numeric(data_cleaned['Total'].str.replace(',', ''), errors='coerce')
    return data_cleaned


def graficar_espiral(data_cleaned):
    """Genera un gráfico 3D de espiral basado en los datos."""
    estados = data_cleaned['Nombre Estado'][:25]
    xs = np.linspace(0, 1, len(estados))
    ys = np.sin(xs * 6 * np.pi)
    zs = np.cos(xs * 6 * np.pi)

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.plot(xs, ys, zs)
    ax.set_xlabel('Índice Normalizado')
    ax.set_ylabel('Función Seno')
    ax.set_zlabel('Función Coseno')
    plt.show()


def graficar_campo_vectorial(data_cleaned):
    """Genera un campo vectorial 3D basado en los datos."""
    estados = data_cleaned['Nombre Estado'][:15]
    x = np.linspace(0, len(estados) - 1, len(estados))
    y = np.linspace(0, len(estados) - 1, len(estados))
    z = np.linspace(0, 10, len(estados))
    X, Y, Z = np.meshgrid(x, y, z)
    U = (X + Y) / 10
    V = (Y - X) / 10
    W = Z / 5

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.quiver(X.flatten(), Y.flatten(), Z.flatten(), U.flatten(), V.flatten(), W.flatten(), length=0.5)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title('Campo Vectorial 3D basado en datos del INEGI')
    plt.show()


def graficar_dispersión(data_cleaned):
    """Genera un gráfico 3D de dispersión basado en los datos."""
    estados = data_cleaned['Nombre Estado'][:25]
    xs = np.arange(len(estados))
    ys = np.arange(len(estados))
    zs = data_cleaned['Total'][:25].values

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(xs, ys, zs, c=zs, cmap='viridis', marker='o')
    ax.set_xlabel('Índice')
    ax.set_ylabel('Estados')
    ax.set_zlabel('Población Total')
    ax.set_title('Gráfico 3D de Dispersión de la Población Total por Estado')
    ax.set_yticks(np.arange(len(estados)))
    ax.set_yticklabels(estados)
    plt.show()


def graficar_tallos(data_cleaned):
    """Genera un gráfico de tallos basado en los datos."""
    estados = data_cleaned['Nombre Estado'][:25]
    x = np.arange(len(estados))
    y = np.zeros_like(x)
    z = data_cleaned['Total'][:25].values

    fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
    ax.stem(x, y, z, basefmt=" ", linefmt="C0-", markerfmt="C0o")
    ax.set_xticks(x)
    ax.set_xticklabels(estados, rotation=90, fontsize=8)
    ax.set_yticklabels([])
    ax.set_title("Gráfico de Tallos de Población Total por Estado")
    ax.set_xlabel("Estados")
    ax.set_ylabel("Eje Y (No utilizado)")
    ax.set_zlabel("Población Total")
    plt.tight_layout()
    plt.show()


def mostrar_menu():
    """Muestra el menú de opciones y ejecuta la opción seleccionada."""
    file_path = 'INEGI_PoblacionTotal_Hombre_Mujeres_clasificados.xlsx'
    data_cleaned = cargar_datos(file_path)

    while True:
        print("\nMenú de Opciones:")
        print("1. Gráfico 3D de Espiral")
        print("2. Campo Vectorial 3D")
        print("3. Gráfico 3D de Dispersión")
        print("4. Gráfico de Tallos")
        print("5. Salir")

        opción = input("Seleccione una opción (1-5): ")

        if opción == '1':
            graficar_espiral(data_cleaned)
        elif opción == '2':
            graficar_campo_vectorial(data_cleaned)
        elif opción == '3':
            graficar_dispersión(data_cleaned)
        elif opción == '4':
            graficar_tallos(data_cleaned)
        elif opción == '5':
            break
        else:
            print("Opción no válida. Inténtelo de nuevo.")


# Ejecutar el menú
mostrar_menu()
