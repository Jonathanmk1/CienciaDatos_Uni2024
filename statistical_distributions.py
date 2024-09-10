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

def grafico_histograma_poblacion(data_cleaned):
    # Seleccionar algunos estados para el ejemplo (25 estados)
    poblacion_total = data_cleaned['Total'][:25]

    # Crear el histograma
    fig, ax = plt.subplots()

    # Histograma con 8 bins
    ax.hist(poblacion_total, bins=8, linewidth=0.5, edgecolor="white", color='skyblue')

    # Configurar los límites y etiquetas
    ax.set(xlim=(poblacion_total.min(), poblacion_total.max()),
           xticks=np.linspace(poblacion_total.min(), poblacion_total.max(), 9),
           ylim=(0, 10),
           yticks=np.arange(0, 11, 2))

    ax.set_title("Distribución de la Población Total por Estado (INEGI)")
    ax.set_xlabel("Población Total")
    ax.set_ylabel("Cantidad de Estados")

    plt.tight_layout()
    plt.show()

def grafico_boxplot_poblacion(data_cleaned):
    # Seleccionar algunos estados para el ejemplo (25 estados)
    poblacion_total = data_cleaned['Total'][:25]

    # Crear categorías de población
    bins = [0, 5e6, 10e6, poblacion_total.max()]  # Limites para las categorías
    labels = ['Baja', 'Media', 'Alta']
    data_cleaned['Grupo Poblacional'] = pd.cut(poblacion_total, bins=bins, labels=labels)

    # Crear los datos para el boxplot
    data_groups = [poblacion_total[data_cleaned['Grupo Poblacional'] == label] for label in labels]

    # Crear el gráfico
    fig, ax = plt.subplots()

    # Graficar con boxplot
    box = ax.boxplot(data_groups, positions=[2, 4, 6], widths=1.5, patch_artist=True,
                     showmeans=False, showfliers=False,
                     medianprops={"color": "white", "linewidth": 0.5},
                     boxprops={"facecolor": "C0", "edgecolor": "white", "linewidth": 0.5},
                     whiskerprops={"color": "C0", "linewidth": 1.5},
                     capprops={"color": "C0", "linewidth": 1.5})

    # Configurar los límites y etiquetas
    ax.set(xlim=(0, 8), xticks=[2, 4, 6], xticklabels=labels,
           ylim=(0, poblacion_total.max() + 1e6), yticks=np.arange(0, poblacion_total.max() + 1e6, 5e6))

    # Etiquetas y título
    ax.set_title("Distribución de la Población por Grupo (INEGI)")
    ax.set_xlabel("Grupo Poblacional")
    ax.set_ylabel("Población Total")

    plt.tight_layout()
    plt.show()

def grafico_poblacion_con_error(data_cleaned):
    # Seleccionar algunos estados para el ejemplo (25 estados)
    estados = data_cleaned['Nombre Estado'][:25]
    poblacion_total = data_cleaned['Total'][:25]

    # Crear los datos para el gráfico
    x = np.arange(len(estados))
    y = poblacion_total.values

    # Simular un error en y (por ejemplo, 10% de la población como error)
    yerr = y * 0.1  # 10% de error

    # Crear el gráfico
    fig, ax = plt.subplots()

    # Graficar con errorbar
    ax.errorbar(x, y, yerr=yerr, fmt='o', linewidth=2, capsize=6, label="Población con error")

    # Configurar los límites y etiquetas
    ax.set(xlim=(-1, len(estados)), xticks=np.arange(len(estados)),
           ylim=(0, poblacion_total.max() + 1e6), yticks=np.arange(0, poblacion_total.max() + 1e6, 5e6))

    # Etiquetas para el eje X con los nombres de los estados
    ax.set_xticks(np.arange(len(estados)))
    ax.set_xticklabels(estados, rotation=90, fontsize=8)

    ax.set_title("Población Total por Estado con Error (INEGI)")
    ax.set_xlabel("Estados")
    ax.set_ylabel("Población Total")

    plt.legend()
    plt.tight_layout()
    plt.show()

def grafico_distribucion_poblacional(data_cleaned):
    # Seleccionar algunos estados para el ejemplo (25 estados)
    poblacion_total = data_cleaned['Total'][:25]

    # Crear categorías de población
    bins = [0, 5e6, 10e6, poblacion_total.max()]  # Limites para las categorías
    labels = ['Baja', 'Media', 'Alta']
    data_cleaned['Grupo Poblacional'] = pd.cut(poblacion_total, bins=bins, labels=labels)

    # Crear los datos para el violinplot
    data_groups = [poblacion_total[data_cleaned['Grupo Poblacional'] == label] for label in labels]

    # Crear el gráfico
    fig, ax = plt.subplots()

    # Graficar con violinplot
    vp = ax.violinplot(data_groups, positions=[2, 4, 6], widths=2,
                       showmeans=False, showmedians=False, showextrema=False)

    # Aplicar transparencia a los cuerpos del violín
    for body in vp['bodies']:
        body.set_alpha(0.9)

    # Configurar los límites y etiquetas
    ax.set(xlim=(0, 8), xticks=[2, 4, 6], xticklabels=labels,
           ylim=(0, poblacion_total.max() + 1e6), yticks=np.arange(0, poblacion_total.max() + 1e6, 5e6))

    # Etiquetas y título
    ax.set_title("Distribución de la Población por Grupo (INEGI)")
    ax.set_xlabel("Grupo Poblacional")
    ax.set_ylabel("Población Total")

    plt.tight_layout()
    plt.show()

def grafico_eventplot(data_cleaned):
    # Seleccionar algunos estados para el ejemplo (25 estados)
    poblacion_total = data_cleaned['Total'][:25]

    # Crear categorías de población
    bins = [0, 5e6, 10e6, poblacion_total.max()]  # Límites para las categorías
    labels = ['Baja', 'Media', 'Alta']
    data_cleaned['Grupo Poblacional'] = pd.cut(poblacion_total, bins=bins, labels=labels)

    # Crear los datos para el eventplot
    data_groups = [poblacion_total[data_cleaned['Grupo Poblacional'] == label].values for label in labels]

    # Asignar posiciones para cada grupo (baja, media, alta)
    positions = [2, 4, 6]

    # Crear el gráfico
    fig, ax = plt.subplots()

    # Graficar con eventplot
    ax.eventplot(data_groups, orientation='vertical', lineoffsets=positions, linewidth=0.75)

    # Configurar los límites y etiquetas
    ax.set(xlim=(0, 8), xticks=positions, xticklabels=labels,
           ylim=(0, poblacion_total.max() + 1e6), yticks=np.arange(0, poblacion_total.max() + 1e6, 5e6))

    # Etiquetas y título
    ax.set_title("Distribución de Eventos de Población por Grupo (INEGI)")
    ax.set_xlabel("Grupo Poblacional")
    ax.set_ylabel("Población Total")

    plt.tight_layout()
    plt.show()

def grafico_hist2d(data_cleaned):
    # Seleccionar algunos estados para el ejemplo (25 estados)
    hombres = data_cleaned['Hombres'][:25].values
    mujeres = data_cleaned['Mujeres'][:25].values

    # Crear el gráfico
    fig, ax = plt.subplots()

    # Graficar con hist2d
    histo = ax.hist2d(hombres, mujeres, bins=30, cmap='Blues')

    # Añadir barra de color para representar la densidad
    plt.colorbar(histo[3], ax=ax)

    # Configurar los límites y etiquetas
    ax.set_xlim(min(hombres), max(hombres))
    ax.set_ylim(min(mujeres), max(mujeres))

    # Etiquetas y título
    ax.set_title("Distribución 2D de Población de Hombres y Mujeres (INEGI)")
    ax.set_xlabel("Población Hombres")
    ax.set_ylabel("Población Mujeres")

    plt.tight_layout()
    plt.show()

def grafico_hexbin(data_cleaned):
    # Seleccionar algunos estados para el ejemplo (25 estados)
    hombres = data_cleaned['Hombres'][:25].values
    mujeres = data_cleaned['Mujeres'][:25].values

    # Crear el gráfico
    fig, ax = plt.subplots()

    # Graficar con hexbin
    hb = ax.hexbin(hombres, mujeres, gridsize=10, cmap='Greens')

    # Añadir barra de color para representar la densidad
    plt.colorbar(hb, ax=ax, label='Cantidad de Estados')

    # Configurar los límites y etiquetas
    ax.set_xlim(min(hombres), max(hombres))
    ax.set_ylim(min(mujeres), max(mujeres))

    # Etiquetas y título
    ax.set_title("Hexbin de Población de Hombres y Mujeres (INEGI)")
    ax.set_xlabel("Población Hombres")
    ax.set_ylabel("Población Mujeres")

    plt.tight_layout()
    plt.show()

def grafico_pie(data_cleaned):
    # Seleccionar algunos estados para el ejemplo (25 estados)
    poblacion_total = data_cleaned['Total'][:25]

    # Crear categorías de población
    bins = [0, 5e6, 10e6, poblacion_total.max()]  # Límites para las categorías
    labels = ['Baja', 'Media', 'Alta']
    data_cleaned['Grupo Poblacional'] = pd.cut(poblacion_total, bins=bins, labels=labels)

    # Calcular la suma de la población por grupo
    grupo_poblacion = data_cleaned.groupby('Grupo Poblacional')['Total'].sum()

    # Datos para el gráfico de pastel
    x = grupo_poblacion.values
    labels = grupo_poblacion.index.tolist()
    colors = plt.get_cmap('Blues')(np.linspace(0.4, 0.8, len(x)))  # Colores personalizados

    # Crear el gráfico
    fig, ax = plt.subplots()

    # Graficar con pie
    ax.pie(x, colors=colors, radius=1, center=(0, 0),
           wedgeprops={"linewidth": 1, "edgecolor": "white"}, labels=labels,
           autopct='%1.1f%%', startangle=140)

    # Añadir un círculo en el centro para convertirlo en un "donut" (opcional)
    centre_circle = plt.Circle((0,0),0.70,fc='white')
    fig.gca().add_artist(centre_circle)

    # Configurar el aspecto del gráfico
    ax.set(aspect="equal")  # Asegura que el pastel sea un círculo
    ax.set_title("Distribución de la Población Total por Grupo (INEGI)", fontsize=14)

    plt.tight_layout()
    plt.show()

def grafico_ecdf(data_cleaned):
    # Seleccionar algunos estados para el ejemplo (25 estados)
    poblacion_total = data_cleaned['Total'][:25].values

    # Ordenar los datos
    x = np.sort(poblacion_total)

    # Crear los valores de la ECDF
    y = np.arange(1, len(x) + 1) / len(x)

    # Crear el gráfico
    fig, ax = plt.subplots()

    ax.step(x, y, where='post')
    ax.set_title("ECDF de la Población Total por Estado (INEGI)")
    ax.set_xlabel("Población Total")
    ax.set_ylabel("ECDF")

    plt.tight_layout()
    plt.show()

def menu():
    """Despliega el menú para seleccionar el tipo de gráfico."""
    print("Seleccione un gráfico:")
    print("1. hist(x)")
    print("2. boxplot(X)")
    print("3. errorbar(x, y, yerr, xerr)")
    print("4. violinplot(D)")
    print("5. eventplot(D)")
    print("6. hist2d(x, y)")
    print("7. hexbin(x, y, C)")
    print("8. pie(x)")
    print("9. ecdf(x)")

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
        grafico_histograma_poblacion(data_cleaned)
    elif opcion == '2':
        grafico_boxplot_poblacion(data_cleaned)
    elif opcion == '3':
        grafico_poblacion_con_error(data_cleaned)
    elif opcion == '4':
        grafico_distribucion_poblacional(data_cleaned)
    elif opcion == '5':
        grafico_eventplot(data_cleaned)
    elif opcion == '6':
        grafico_hist2d(data_cleaned)
    elif opcion == '7':
        grafico_hexbin(data_cleaned)
    elif opcion == '8':
        grafico_pie(data_cleaned)
    elif opcion == '9':
        grafico_ecdf(data_cleaned)
    else:
        print("Opción no válida. Por favor, selecciona un número del 1 al 4.")

if __name__ == "__main__":
    main()
