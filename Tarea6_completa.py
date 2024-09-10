import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# Cargar y limpiar los datos
def cargar_datos():
    file_path = 'INEGI_PoblacionTotal_Hombre_Mujeres_clasificados.xlsx'
    data = pd.read_excel(file_path, engine='openpyxl')
    data_cleaned = data.drop([0, 1])  # Eliminar las dos primeras filas innecesarias
    data_cleaned.columns = ['Estado', 'Nombre Estado', 'Total', 'Hombres', 'Mujeres']
    data_cleaned['Total'] = pd.to_numeric(data_cleaned['Total'].str.replace(',', ''), errors='coerce')
    return data_cleaned

def grafico_poblacion_total():
    data_cleaned = cargar_datos()  # Llamar a cargar_datos para obtener los datos limpios

    # Seleccionar algunos estados para el ejemplo (25 estados)
    estados = data_cleaned['Nombre Estado'][:25]
    poblacion_total = data_cleaned['Total'][:25]

    # Crear los datos para el gráfico
    x = np.linspace(0, len(estados)-1, 100)
    y = np.interp(x, np.arange(len(estados)), poblacion_total)

    # Generar puntos adicionales para mostrar en el gráfico
    x2 = np.arange(len(estados))
    y2 = poblacion_total.values

    # Crear el gráfico
    fig, ax = plt.subplots()

    ax.plot(x2, y2 + 2.5e6, 'x', markeredgewidth=2, label='Población desplazada +2.5M')
    ax.plot(x, y, linewidth=2.0, label='Población Total')
    ax.plot(x2, y2 - 2.5e6, 'o-', linewidth=2, label='Población desplazada -2.5M')

    # Configurar los límites y etiquetas
    ax.set(xlim=(0, 24), xticks=np.arange(0, 25, 5),
           ylim=(0, 15e6), yticks=np.arange(0, 15e6, 5e6))

    # Etiquetas para el eje X con los nombres de los estados
    ax.set_xticks(np.arange(len(estados)))
    ax.set_xticklabels(estados, rotation=90, fontsize=8)

    ax.set_title("Población Total por Estado (INEGI)")
    ax.set_xlabel("Estados")
    ax.set_ylabel("Población Total")

    plt.legend()
    plt.tight_layout()
    plt.show()

def grafico_area_rellena(estados, poblacion_total):
    """Genera un gráfico de área rellena entre dos líneas."""
    x = np.arange(len(estados))
    y1 = poblacion_total.values
    y2 = y1 * np.random.uniform(0.95, 1.05, len(y1))  # Población proyectada o variada

    fig, ax = plt.subplots()

    # Rellenar el área entre y1 y y2
    ax.fill_between(x, y1, y2, alpha=0.5, color='skyblue', label='Intervalo de población')

    # Graficar la línea media
    ax.plot(x, (y1 + y2) / 2, linewidth=2, color='blue', label='Población Media')

    # Configurar los límites y etiquetas
    ax.set(xlim=(-1, len(estados)), xticks=np.arange(len(estados)),
           ylim=(0, max(y1.max(), y2.max()) + 1e6), yticks=np.arange(0, max(y1.max(), y2.max()) + 1e6, 5e6))

    # Etiquetas para el eje X con los nombres de los estados
    ax.set_xticks(np.arange(len(estados)))
    ax.set_xticklabels(estados, rotation=90, fontsize=8)

    ax.set_title("Rango de Población por Estado (INEGI)")
    ax.set_xlabel("Estados")
    ax.set_ylabel("Población Total")

    plt.legend()
    plt.tight_layout()
    plt.show()

def grafico_apilado_hombres_mujeres_otros(estados, hombres, mujeres, poblacion_total):
    """Genera un gráfico apilado de hombres, mujeres y otros."""
    x = np.arange(len(estados))
    y = np.vstack([hombres, mujeres, poblacion_total - hombres - mujeres])

    fig, ax = plt.subplots()

    # Graficar con stackplot
    ax.stackplot(x, y, labels=['Hombres', 'Mujeres', 'Otros'], alpha=0.8)

    # Configurar los límites y etiquetas
    ax.set(xlim=(-1, len(estados)), xticks=np.arange(len(estados)),
           ylim=(0, poblacion_total.max() + 1e6), yticks=np.arange(0, poblacion_total.max() + 1e6, 5e6))

    # Etiquetas para el eje X con los nombres de los estados
    ax.set_xticks(np.arange(len(estados)))
    ax.set_xticklabels(estados, rotation=90, fontsize=8)

    ax.set_title("Población Total por Estado (INEGI) - Apilado")
    ax.set_xlabel("Estados")
    ax.set_ylabel("Población Total")

    # Añadir la leyenda
    ax.legend(loc='upper left')

    plt.tight_layout()
    plt.show()

def grafico_step(estados, poblacion_total):
    """Genera un gráfico de líneas con el método step."""
    x = np.arange(len(estados))
    y = poblacion_total.values

    fig, ax = plt.subplots()

    # Graficar con step
    ax.step(x, y, linewidth=2.5, where='mid')

    # Configurar los límites y etiquetas
    ax.set(xlim=(-1, len(estados)), xticks=np.arange(len(estados)),
           ylim=(0, poblacion_total.max() + 1e6), yticks=np.arange(0, poblacion_total.max() + 1e6, 5e6))

    # Etiquetas para el eje X con los nombres de los estados
    ax.set_xticks(np.arange(len(estados)))
    ax.set_xticklabels(estados, rotation=90, fontsize=8)

    ax.set_title("Población Total por Estado (INEGI)")
    ax.set_xlabel("Estados")
    ax.set_ylabel("Población Total")

    plt.tight_layout()
    plt.show()


# Menú principal
def menu():
    print("Seleccione un gráfico:")
    print("1. plot(x, y)")
    print("2. scatter(x, y)")
    print("3. bar(x, height)")
    print("4. stem(x, y)")
    print("5. fill_between(x, y1, y2)")
    print("6. stackplot(x, y)")
    # Añadir más opciones para otros gráficos

    opcion = int(input("Ingrese el número de la opción deseada: "))

    if opcion == 1:
        grafico_poblacion_total()
    elif opcion == 2:
        scatter_plot()
    elif opcion == 3:
        bar_plot()
    # Agregar más elif para otros gráficos
    else:
        print("Opción no válida")


# Llamar al menú
if __name__ == "__main__":
    menu()
