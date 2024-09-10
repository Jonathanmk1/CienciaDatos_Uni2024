import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Cargar el archivo de Excel
file_path = 'INEGI_PoblacionTotal_Hombre_Mujeres_clasificados.xlsx'
data = pd.read_excel(file_path, engine='openpyxl')  # Especificar el motor 'openpyxl'

# Limpiar los datos
data_cleaned = data.drop([0, 1])  # Eliminar las dos primeras filas innecesarias
data_cleaned.columns = ['Estado', 'Nombre Estado', 'Total', 'Hombres', 'Mujeres']
data_cleaned['Total'] = pd.to_numeric(data_cleaned['Total'].str.replace(',', ''), errors='coerce')

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

##########################################################################
# Cargar el archivo de Excel
file_path = 'INEGI_PoblacionTotal_Hombre_Mujeres_clasificados.xlsx'
data = pd.read_excel(file_path, engine='openpyxl')  # Especificar el motor 'openpyxl'

# Limpiar los datos
data_cleaned = data.drop([0, 1])  # Eliminar las dos primeras filas innecesarias
data_cleaned.columns = ['Estado', 'Nombre Estado', 'Total', 'Hombres', 'Mujeres']
data_cleaned['Total'] = pd.to_numeric(data_cleaned['Total'].str.replace(',', ''), errors='coerce')

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
###########################################################################333

# Cargar el archivo de Excel
file_path = 'INEGI_PoblacionTotal_Hombre_Mujeres_clasificados.xlsx'
data = pd.read_excel(file_path, engine='openpyxl')  # Especificar el motor 'openpyxl'

# Limpiar los datos
data_cleaned = data.drop([0, 1])  # Eliminar las dos primeras filas innecesarias
data_cleaned.columns = ['Estado', 'Nombre Estado', 'Total', 'Hombres', 'Mujeres']
data_cleaned['Total'] = pd.to_numeric(data_cleaned['Total'].str.replace(',', ''), errors='coerce')

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

# Graficar con bar
bars = ax.bar(x, y, width=widths, color=plt.cm.viridis(np.interp(colors, (15, 80), (0, 1))), edgecolor="white", linewidth=0.7)

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

############################################################3

# Cargar el archivo de Excel
file_path = 'INEGI_PoblacionTotal_Hombre_Mujeres_clasificados.xlsx'
data = pd.read_excel(file_path, engine='openpyxl')  # Especificar el motor 'openpyxl'

# Limpiar los datos
data_cleaned = data.drop([0, 1])  # Eliminar las dos primeras filas innecesarias
data_cleaned.columns = ['Estado', 'Nombre Estado', 'Total', 'Hombres', 'Mujeres']
data_cleaned['Total'] = pd.to_numeric(data_cleaned['Total'].str.replace(',', ''), errors='coerce')

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
#####################################################
# Cargar el archivo de Excel
file_path = 'INEGI_PoblacionTotal_Hombre_Mujeres_clasificados.xlsx'
data = pd.read_excel(file_path, engine='openpyxl')  # Especificar el motor 'openpyxl'

# Limpiar los datos
data_cleaned = data.drop([0, 1])  # Eliminar las dos primeras filas innecesarias
data_cleaned.columns = ['Estado', 'Nombre Estado', 'Total', 'Hombres', 'Mujeres']
data_cleaned['Total'] = pd.to_numeric(data_cleaned['Total'].str.replace(',', ''), errors='coerce')

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
"PRIMERA PARTE DE CH"
###############################################################

# Cargar el archivo de Excel
file_path = 'INEGI_PoblacionTotal_Hombre_Mujeres_clasificados.xlsx'
data = pd.read_excel(file_path, engine='openpyxl')  # Especificar el motor 'openpyxl'

# Limpiar los datos
data_cleaned = data.drop([0, 1])  # Eliminar las dos primeras filas innecesarias
data_cleaned.columns = ['Estado', 'Nombre Estado', 'Total', 'Hombres', 'Mujeres']

# Convertir las columnas a numéricas
data_cleaned['Total'] = pd.to_numeric(data_cleaned['Total'].str.replace(',', ''), errors='coerce')
data_cleaned['Hombres'] = pd.to_numeric(data_cleaned['Hombres'].str.replace(',', ''), errors='coerce')
data_cleaned['Mujeres'] = pd.to_numeric(data_cleaned['Mujeres'].str.replace(',', ''), errors='coerce')

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
################################################3
# Cargar el archivo de Excel
file_path = 'INEGI_PoblacionTotal_Hombre_Mujeres_clasificados.xlsx'
data = pd.read_excel(file_path, engine='openpyxl')  # Especificar el motor 'openpyxl'

# Limpiar los datos
data_cleaned = data.drop([0, 1])  # Eliminar las dos primeras filas innecesarias
data_cleaned.columns = ['Estado', 'Nombre Estado', 'Total', 'Hombres', 'Mujeres']
data_cleaned['Total'] = pd.to_numeric(data_cleaned['Total'].str.replace(',', ''), errors='coerce')

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

###################################################################

# Cargar el archivo de Excel
file_path = 'INEGI_PoblacionTotal_Hombre_Mujeres_clasificados.xlsx'
data = pd.read_excel(file_path, engine='openpyxl')  # Especificar el motor 'openpyxl'

# Limpiar los datos
data_cleaned = data.drop([0, 1])  # Eliminar las dos primeras filas innecesarias
data_cleaned.columns = ['Estado', 'Nombre Estado', 'Total', 'Hombres', 'Mujeres']
data_cleaned['Total'] = pd.to_numeric(data_cleaned['Total'].str.replace(',', ''), errors='coerce')

# Seleccionar algunos estados para el ejemplo (25 estados)
poblacion_total = data_cleaned['Total'][:25]

# Crear el histograma
fig, ax = plt.subplots()

# Histograma con 8 bins
ax.hist(poblacion_total, bins=8, linewidth=0.5, edgecolor="white", color='skyblue')

# Configurar los límites y etiquetas
ax.set(xlim=(poblacion_total.min(), poblacion_total.max()), xticks=np.linspace(poblacion_total.min(), poblacion_total.max(), 9),
       ylim=(0, 10), yticks=np.arange(0, 11, 2))

ax.set_title("Distribución de la Población Total por Estado (INEGI)")
ax.set_xlabel("Población Total")
ax.set_ylabel("Cantidad de Estados")

plt.tight_layout()
plt.show()

###############################################################333

# Cargar el archivo de Excel
file_path = 'INEGI_PoblacionTotal_Hombre_Mujeres_clasificados.xlsx'
data = pd.read_excel(file_path, engine='openpyxl')  # Especificar el motor 'openpyxl'

# Limpiar los datos
data_cleaned = data.drop([0, 1])  # Eliminar las dos primeras filas innecesarias
data_cleaned.columns = ['Estado', 'Nombre Estado', 'Total', 'Hombres', 'Mujeres']
data_cleaned['Total'] = pd.to_numeric(data_cleaned['Total'].str.replace(',', ''), errors='coerce')

# Seleccionar algunos estados para el ejemplo (25 estados)
poblacion_total = data_cleaned['Total'][:25]

# Crear categorías de población
# Dividimos los datos de población en tres grupos (por ejemplo: baja, media, alta)
bins = [0, 5e6, 10e6, poblacion_total.max()]  # Limites para las categorías
labels = ['Baja', 'Media', 'Alta']
data_cleaned['Grupo Poblacional'] = pd.cut(poblacion_total, bins=bins, labels=labels)

# Crear los datos para el boxplot
# Agrupamos por el grupo poblacional
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
############################################################################

# Cargar el archivo de Excel
file_path = 'INEGI_PoblacionTotal_Hombre_Mujeres_clasificados.xlsx'
data = pd.read_excel(file_path, engine='openpyxl')  # Especificar el motor 'openpyxl'

# Limpiar los datos
data_cleaned = data.drop([0, 1])  # Eliminar las dos primeras filas innecesarias
data_cleaned.columns = ['Estado', 'Nombre Estado', 'Total', 'Hombres', 'Mujeres']
data_cleaned['Total'] = pd.to_numeric(data_cleaned['Total'].str.replace(',', ''), errors='coerce')

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
#################################################

# Cargar el archivo de Excel
file_path = 'INEGI_PoblacionTotal_Hombre_Mujeres_clasificados.xlsx'
data = pd.read_excel(file_path, engine='openpyxl')  # Especificar el motor 'openpyxl'

# Limpiar los datos
data_cleaned = data.drop([0, 1])  # Eliminar las dos primeras filas innecesarias
data_cleaned.columns = ['Estado', 'Nombre Estado', 'Total', 'Hombres', 'Mujeres']
data_cleaned['Total'] = pd.to_numeric(data_cleaned['Total'].str.replace(',', ''), errors='coerce')

# Seleccionar algunos estados para el ejemplo (25 estados)
poblacion_total = data_cleaned['Total'][:25]

# Crear categorías de población
# Dividimos los datos de población en tres grupos (por ejemplo: baja, media, alta)
bins = [0, 5e6, 10e6, poblacion_total.max()]  # Limites para las categorías
labels = ['Baja', 'Media', 'Alta']
data_cleaned['Grupo Poblacional'] = pd.cut(poblacion_total, bins=bins, labels=labels)

# Crear los datos para el violinplot
# Agrupamos por el grupo poblacional
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
###################################################3

# Cargar el archivo de Excel
file_path = 'INEGI_PoblacionTotal_Hombre_Mujeres_clasificados.xlsx'
data = pd.read_excel(file_path, engine='openpyxl')  # Especificar el motor 'openpyxl'

# Limpiar los datos
data_cleaned = data.drop([0, 1])  # Eliminar las dos primeras filas innecesarias
data_cleaned.columns = ['Estado', 'Nombre Estado', 'Total', 'Hombres', 'Mujeres']
data_cleaned['Total'] = pd.to_numeric(data_cleaned['Total'].str.replace(',', ''), errors='coerce')

# Seleccionar algunos estados para el ejemplo (25 estados)
poblacion_total = data_cleaned['Total'][:25]

# Crear categorías de población
# Dividimos los datos de población en tres grupos (por ejemplo: baja, media, alta)
bins = [0, 5e6, 10e6, poblacion_total.max()]  # Límites para las categorías
labels = ['Baja', 'Media', 'Alta']
data_cleaned['Grupo Poblacional'] = pd.cut(poblacion_total, bins=bins, labels=labels)

# Crear los datos para el eventplot
# Agrupamos por el grupo poblacional
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
#######################################################3

# Cargar el archivo de Excel
file_path = 'INEGI_PoblacionTotal_Hombre_Mujeres_clasificados.xlsx'
data = pd.read_excel(file_path, engine='openpyxl')  # Especificar el motor 'openpyxl'

# Limpiar los datos
data_cleaned = data.drop([0, 1])  # Eliminar las dos primeras filas innecesarias
data_cleaned.columns = ['Estado', 'Nombre Estado', 'Total', 'Hombres', 'Mujeres']
data_cleaned['Hombres'] = pd.to_numeric(data_cleaned['Hombres'].str.replace(',', ''), errors='coerce')
data_cleaned['Mujeres'] = pd.to_numeric(data_cleaned['Mujeres'].str.replace(',', ''), errors='coerce')

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
######################################################33

# Cargar el archivo de Excel
file_path = 'INEGI_PoblacionTotal_Hombre_Mujeres_clasificados.xlsx'
data = pd.read_excel(file_path, engine='openpyxl')  # Especificar el motor 'openpyxl'

# Limpiar los datos
data_cleaned = data.drop([0, 1])  # Eliminar las dos primeras filas innecesarias
data_cleaned.columns = ['Estado', 'Nombre Estado', 'Total', 'Hombres', 'Mujeres']
data_cleaned['Hombres'] = pd.to_numeric(data_cleaned['Hombres'].str.replace(',', ''), errors='coerce')
data_cleaned['Mujeres'] = pd.to_numeric(data_cleaned['Mujeres'].str.replace(',', ''), errors='coerce')

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
#############################################################3
# Estilo del gráfico
plt.style.use('_mpl-gallery-nogrid')

# Cargar el archivo de Excel
file_path = 'INEGI_PoblacionTotal_Hombre_Mujeres_clasificados.xlsx'
data = pd.read_excel(file_path, engine='openpyxl')  # Especificar el motor 'openpyxl'

# Limpiar los datos
data_cleaned = data.drop([0, 1])  # Eliminar las dos primeras filas innecesarias
data_cleaned.columns = ['Estado', 'Nombre Estado', 'Total', 'Hombres', 'Mujeres']
data_cleaned['Total'] = pd.to_numeric(data_cleaned['Total'].str.replace(',', ''), errors='coerce')

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
###############################################

# Cargar el archivo de Excel
file_path = 'INEGI_PoblacionTotal_Hombre_Mujeres_clasificados.xlsx'
data = pd.read_excel(file_path, engine='openpyxl')  # Especificar el motor 'openpyxl'

# Limpiar los datos
data_cleaned = data.drop([0, 1])  # Eliminar las dos primeras filas innecesarias
data_cleaned.columns = ['Estado', 'Nombre Estado', 'Total', 'Hombres', 'Mujeres']
data_cleaned['Total'] = pd.to_numeric(data_cleaned['Total'].str.replace(',', ''), errors='coerce')

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

##############################################33

# Cargar el archivo de Excel
file_path = 'INEGI_PoblacionTotal_Hombre_Mujeres_clasificados.xlsx'
data = pd.read_excel(file_path, engine='openpyxl')  # Especificar el motor 'openpyxl'

# Limpiar los datos
data_cleaned = data.drop([0, 1])  # Eliminar las dos primeras filas innecesarias
data_cleaned.columns = ['Estado', 'Nombre Estado', 'Total', 'Hombres', 'Mujeres']
data_cleaned['Total'] = pd.to_numeric(data_cleaned['Total'].str.replace(',', ''), errors='coerce')

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
##################################################3

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
X = np.linspace(0, matrix_size - 1, matrix_size)
Y = np.linspace(0, matrix_size - 1, matrix_size)
X, Y = np.meshgrid(X, Y)

# Ajustar los datos de población para que encajen en la malla
# Generar datos Z basados en una función o usando los datos de población
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
#########################################3

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

############################################################

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
###########################################################3

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

##################################################3
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

#########################################

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
##################################

# Cargar el archivo de Excel
file_path = 'INEGI_PoblacionTotal_Hombre_Mujeres_clasificados.xlsx'
data = pd.read_excel(file_path, engine='openpyxl')  # Especificar el motor 'openpyxl'

# Limpiar los datos
data_cleaned = data.drop([0, 1])  # Eliminar las dos primeras filas innecesarias
data_cleaned.columns = ['Estado', 'Nombre Estado', 'Total', 'Hombres', 'Mujeres']
data_cleaned['Total'] = pd.to_numeric(data_cleaned['Total'].str.replace(',', ''), errors='coerce')

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
###################################3

























# Cargar el archivo de Excel
file_path = 'INEGI_PoblacionTotal_Hombre_Mujeres_clasificados.xlsx'
data = pd.read_excel(file_path, engine='openpyxl')  # Especificar el motor 'openpyxl'

# Limpiar los datos
data_cleaned = data.drop([0, 1])  # Eliminar las dos primeras filas innecesarias
data_cleaned.columns = ['Estado', 'Nombre Estado', 'Total', 'Hombres', 'Mujeres']
data_cleaned['Total'] = pd.to_numeric(data_cleaned['Total'].str.replace(',', ''), errors='coerce')

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
###########################################

# Cargar el archivo de Excel
file_path = 'INEGI_PoblacionTotal_Hombre_Mujeres_clasificados.xlsx'
data = pd.read_excel(file_path, engine='openpyxl')  # Especificar el motor 'openpyxl'

# Limpiar los datos
data_cleaned = data.drop([0, 1])  # Eliminar las dos primeras filas innecesarias
data_cleaned.columns = ['Estado', 'Nombre Estado', 'Total', 'Hombres', 'Mujeres']
data_cleaned['Total'] = pd.to_numeric(data_cleaned['Total'].str.replace(',', ''), errors='coerce')

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
####################################

# Cargar el archivo de Excel
file_path = 'INEGI_PoblacionTotal_Hombre_Mujeres_clasificados.xlsx'
data = pd.read_excel(file_path, engine='openpyxl')  # Especificar el motor 'openpyxl'

# Limpiar los datos
data_cleaned = data.drop([0, 1])  # Eliminar las dos primeras filas innecesarias
data_cleaned.columns = ['Estado', 'Nombre Estado', 'Total', 'Hombres', 'Mujeres']
data_cleaned['Total'] = pd.to_numeric(data_cleaned['Total'].str.replace(',', ''), errors='coerce')

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
############################################

# Cargar datos del archivo Excel
file_path = 'INEGI_PoblacionTotal_Hombre_Mujeres_clasificados.xlsx'
data = pd.read_excel(file_path, engine='openpyxl')

# Limpiar datos
data_cleaned = data.drop([0, 1])  # Eliminar las dos primeras filas innecesarias
data_cleaned.columns = ['Estado', 'Nombre Estado', 'Total', 'Hombres', 'Mujeres']
data_cleaned['Total'] = pd.to_numeric(data_cleaned['Total'].str.replace(',', ''), errors='coerce')

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
########################################





















# Cargar el archivo de Excel
file_path = 'INEGI_PoblacionTotal_Hombre_Mujeres_clasificados.xlsx'
data = pd.read_excel(file_path, engine='openpyxl')  # Especificar el motor 'openpyxl'

# Limpiar los datos
data_cleaned = data.drop([0, 1])  # Eliminar las dos primeras filas innecesarias
data_cleaned.columns = ['Estado', 'Nombre Estado', 'Total', 'Hombres', 'Mujeres']
data_cleaned['Total'] = pd.to_numeric(data_cleaned['Total'].str.replace(',', ''), errors='coerce')

# Seleccionar algunos estados para el ejemplo (25 estados)
estados = data_cleaned['Nombre Estado'][:25]
poblacion_total = data_cleaned['Total'][:25]

# Generar datos para el gráfico
n = len(estados)
xs = np.linspace(0, 1, n)  # Eje X: valores normalizados
ys = np.sin(xs * 6 * np.pi)  # Eje Y: función seno para espiral
zs = np.cos(xs * 6 * np.pi)  # Eje Z: función coseno para espiral

# Plot
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.plot(xs, ys, zs)

# Configurar etiquetas de los ejes
ax.set_xlabel('Índice Normalizado')
ax.set_ylabel('Función Seno')
ax.set_zlabel('Función Coseno')

plt.show()
######################################################
# Cargar el archivo de Excel
file_path = 'INEGI_PoblacionTotal_Hombre_Mujeres_clasificados.xlsx'
data = pd.read_excel(file_path, engine='openpyxl')

# Limpiar los datos
data_cleaned = data.drop([0, 1])  # Eliminar las dos primeras filas innecesarias
data_cleaned.columns = ['Estado', 'Nombre Estado', 'Total', 'Hombres', 'Mujeres']
data_cleaned['Total'] = pd.to_numeric(data_cleaned['Total'].str.replace(',', ''), errors='coerce')

# Seleccionar algunos estados para el ejemplo (15 estados)
estados = data_cleaned['Nombre Estado'][:15]
poblacion_total = data_cleaned['Total'][:15]

# Generar malla 3D
n = len(estados)
x = np.linspace(0, n-1, n)
y = np.linspace(0, n-1, n)
z = np.linspace(0, 10, n)  # Ajustamos el rango para la visualización

X, Y, Z = np.meshgrid(x, y, z)

# Usar población total para crear U, V y W (valores ficticios para ilustrar)
U = (X + Y)/10
V = (Y - X)/10
W = Z/5

# Plot
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.quiver(X.flatten(), Y.flatten(), Z.flatten(), U.flatten(), V.flatten(), W.flatten(), length=0.5)

ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
ax.set_title('Campo Vectorial 3D basado en datos del INEGI')

plt.show()
#################################################

# Cargar el archivo de Excel
file_path = 'INEGI_PoblacionTotal_Hombre_Mujeres_clasificados.xlsx'
data = pd.read_excel(file_path, engine='openpyxl')  # Especificar el motor 'openpyxl'

# Limpiar los datos
data_cleaned = data.drop([0, 1])  # Eliminar las dos primeras filas innecesarias
data_cleaned.columns = ['Estado', 'Nombre Estado', 'Total', 'Hombres', 'Mujeres']
data_cleaned['Total'] = pd.to_numeric(data_cleaned['Total'].str.replace(',', ''), errors='coerce')

# Seleccionar algunos estados para el ejemplo (25 estados)
estados = data_cleaned['Nombre Estado'][:25]
poblacion_total = data_cleaned['Total'][:25]

# Generar datos para el gráfico 3D de dispersión
xs = np.arange(len(estados))  # Usar un rango de valores para xs
ys = np.arange(len(estados))  # Usar el índice de los estados para ys
zs = poblacion_total.values  # Población total como zs

# Crear el gráfico
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

ax.scatter(xs, ys, zs, c=zs, cmap='viridis', marker='o')

# Configurar etiquetas y título
ax.set_xlabel('Índice')
ax.set_ylabel('Estados')
ax.set_zlabel('Población Total')
ax.set_title('Gráfico 3D de Dispersión de la Población Total por Estado')

# Configurar ticks en el eje y con nombres de estados
ax.set_yticks(np.arange(len(estados)))
ax.set_yticklabels(estados)

plt.show()
###########################################

plt.style.use('_mpl-gallery')

# Cargar el archivo de Excel
file_path = 'INEGI_PoblacionTotal_Hombre_Mujeres_clasificados.xlsx'
data = pd.read_excel(file_path, engine='openpyxl')  # Especificar el motor 'openpyxl'

# Limpiar los datos
data_cleaned = data.drop([0, 1])  # Eliminar las dos primeras filas innecesarias
data_cleaned.columns = ['Estado', 'Nombre Estado', 'Total', 'Hombres', 'Mujeres']
data_cleaned['Total'] = pd.to_numeric(data_cleaned['Total'].str.replace(',', ''), errors='coerce')

# Seleccionar algunos estados para el ejemplo (25 estados)
estados = data_cleaned['Nombre Estado'][:25]
poblacion_total = data_cleaned['Total'][:25]

# Datos para el gráfico
x = np.arange(len(estados))  # Índices de los estados
y = np.zeros_like(x)  # En este caso, podemos usar 0 para representar los valores en el eje y
z = poblacion_total.values  # Población total

# Crear el gráfico
fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
ax.stem(x, y, z, basefmt=" ", linefmt="C0-", markerfmt="C0o")

# Etiquetas
ax.set_xticks(x)
ax.set_xticklabels(estados, rotation=90, fontsize=8)
ax.set_yticklabels([])  # No necesitamos etiquetas para el eje y en este caso

# Configurar los límites y etiquetas
ax.set_title("Gráfico de Tallos de Población Total por Estado")
ax.set_xlabel("Estados")
ax.set_ylabel("Eje Y (No utilizado)")
ax.set_zlabel("Población Total")

plt.tight_layout()
plt.show()
####################################################333