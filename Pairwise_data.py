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

def grafico_poblacion_desplazada(data_cleaned, desplazamiento=2.5e6):
    """
    Genera un gráfico con la población total por estado y líneas desplazadas.

    Parámetros:
    - data_cleaned: DataFrame con los datos ya limpiados.
    - desplazamiento: Cantidad en la que se desplazan los puntos para las gráficas adicionales.
    """

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

    # Graficar la población total y las líneas desplazadas
    ax.plot(x2, y2 + desplazamiento, 'x', markeredgewidth=2, label=f'Población desplazada +{desplazamiento/1e6}M')
    ax.plot(x, y, linewidth=2.0, label='Población Total')
    ax.plot(x2, y2 - desplazamiento, 'o-', linewidth=2, label=f'Población desplazada -{desplazamiento/1e6}M')

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

def grafico_scatter_poblacion(data_cleaned):
    # Seleccionar algunos estados para el ejemplo (25 estados)
    estados = data_cleaned['Nombre Estado'][:25]
    poblacion_total = data_cleaned['Total'][:25]

    # Crear los datos para el gráfico
    x = np.arange(len(estados))
    y = poblacion_total.values

    # Tamaño y color basados en la población
    sizes = np.interp(y, (y.min(), y.max()), (15, 80))  # Tamaño de los puntos
    colors = np.interp(y, (y.min(), y.max()), (15, 80))  # Color de los puntos

    # Crear el gráfico
    fig, ax = plt.subplots()

    # Graficar con scatter
    scatter = ax.scatter(x, y, s=sizes, c=colors, cmap='viridis', vmin=0, vmax=80)

    # Configurar los límites y etiquetas
    ax.set(xlim=(-1, len(estados)), xticks=np.arange(len(estados)),
           ylim=(0, poblacion_total.max() + 1e6), yticks=np.arange(0, poblacion_total.max() + 1e6, 5e6))

    # Etiquetas para el eje X con los nombres de los estados
    ax.set_xticks(np.arange(len(estados)))
    ax.set_xticklabels(estados, rotation=90, fontsize=8)

    ax.set_title("Población Total por Estado (INEGI)")
    ax.set_xlabel("Estados")
    ax.set_ylabel("Población Total")

    # Agregar una barra de colores
    cbar = plt.colorbar(scatter, ax=ax)
    cbar.set_label('Tamaño/Color basado en la población')

    plt.tight_layout()
    plt.show()

def grafico_barras_poblacion(data_cleaned):
    # Seleccionar algunos estados para el ejemplo (25 estados)
    estados = data_cleaned['Nombre Estado'][:25]
    poblacion_total = data_cleaned['Total'][:25]

    # Crear los datos para el gráfico
    x = np.arange(len(estados))
    y = poblacion_total.values

    # Tamaño y color de las barras basados en la población
    widths = np.interp(y, (y.min(), y.max()), (0.5, 1.0))  # Ancho de las barras
    colors = np.interp(y, (y.min(), y.max()), (15, 80))  # Color de las barras

    # Crear el gráfico
    fig, ax = plt.subplots()

    # Graficar con barras
    bars = ax.bar(x, y, width=widths, color=plt.cm.viridis(np.interp(colors, (15, 80), (0, 1))),
                  edgecolor="white", linewidth=0.7)

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

def grafico_stem_poblacion(data_cleaned):
    # Seleccionar algunos estados para el ejemplo (25 estados)
    estados = data_cleaned['Nombre Estado'][:25]
    poblacion_total = data_cleaned['Total'][:25]

    # Crear los datos para el gráfico
    x = np.arange(len(estados))
    y = poblacion_total.values

    # Crear el gráfico
    fig, ax = plt.subplots()

    # Graficar con stem
    markers, stems, baseline = ax.stem(x, y, linefmt='-', markerfmt='o', basefmt=" ")

    # Normalizar colores entre 0 y 1 para cmap
    norm = plt.Normalize(vmin=y.min(), vmax=y.max())
    cmap = plt.get_cmap('viridis')  # Mapa de colores

    # Ajustar el color de los tallos
    stem_color = cmap(norm(y.mean()))  # Usar el color promedio para los tallos
    stems.set_color(stem_color)

    # Ajustar el color de los marcadores
    for i in range(len(x)):
        color = cmap(norm(y[i]))
        markers.set_markerfacecolor(color)
        markers.set_markeredgecolor('black')
        markers.set_markeredgewidth(1)  # Ajusta el grosor del borde del marcador si es necesario

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

def grafico_rango_poblacion(data_cleaned):
    # Seleccionar algunos estados para el ejemplo (25 estados)
    estados = data_cleaned['Nombre Estado'][:25]
    poblacion_total = data_cleaned['Total'][:25]

    # Crear los datos para el gráfico
    x = np.arange(len(estados))
    y1 = poblacion_total.values
    y2 = y1 * np.random.uniform(0.95, 1.05, len(y1))  # Población proyectada o variada

    # Crear el gráfico
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

def grafico_poblacion_apilado(data_cleaned):
    # Seleccionar algunos estados para el ejemplo (25 estados)
    estados = data_cleaned['Nombre Estado'][:25]
    poblacion_total = data_cleaned['Total'][:25]
    hombres = data_cleaned['Hombres'][:25]
    mujeres = data_cleaned['Mujeres'][:25]

    # Crear los datos para el gráfico
    x = np.arange(len(estados))
    y = np.vstack([hombres, mujeres, poblacion_total - hombres - mujeres])

    # Crear el gráfico
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

def grafico_poblacion_step(data_cleaned):
    # Seleccionar algunos estados para el ejemplo (25 estados)
    estados = data_cleaned['Nombre Estado'][:25]
    poblacion_total = data_cleaned['Total'][:25]

    # Crear los datos para el gráfico
    x = np.arange(len(estados))
    y = poblacion_total.values

    # Crear el gráfico
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


def menu():
    """Despliega el menú para seleccionar el tipo de gráfico."""
    print("Seleccione un gráfico:")
    print("1. plot(x, y)")
    print("2. scatter(x, y)")
    print("3. bar(x, height)")
    print("4. stem(x, y)")
    print("5. fill_between(x, y1, y2)")
    print("6. stackplot(x, y)")
    print("7. stairs(values)")

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
        grafico_poblacion_desplazada(data_cleaned)
    elif opcion == '2':
        grafico_scatter_poblacion(data_cleaned)
    elif opcion == '3':
        grafico_barras_poblacion(data_cleaned)
    elif opcion == '4':
        grafico_stem_poblacion(data_cleaned)
    elif opcion == '5':
        grafico_rango_poblacion(data_cleaned)
    elif opcion == '6':
        grafico_poblacion_apilado(data_cleaned)
    elif opcion == '7':
        grafico_poblacion_step(data_cleaned)
    else:
        print("Opción no válida. Por favor, selecciona un número del 1 al 4.")

if __name__ == "__main__":
    main()
