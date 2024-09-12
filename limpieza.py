"""
Le pregunte al chat:
Hacer limpieza a este archivo, tengo un codigo como apoyo pero no me da resultados que deseo:

Puedo realizar la limpieza de los datos que has proporcionado directamente sobre el archivo que has subido. A continuación, cargaré el archivo, realizaré las mismas operaciones que describes en tu código, y te mostraré los resultados que esperas.

Voy a proceder a leer el archivo y limpiar los datos.


"""
import pandas as pd
import gzip


# Función para cargar y limpiar datos
def cargar_y_limpiar_datos(archivo_comprimido, nombres_columnas):
    # Abrir archivo comprimido
    with gzip.open(archivo_comprimido, 'rt', encoding='utf-8') as archivo:
        # Leer archivo CSV sin encabezados y asignar nombres de columnas
        datos = pd.read_csv(archivo, header=None)
        datos.columns = nombres_columnas

    # Eliminar filas duplicadas
    datos_limpios = datos.drop_duplicates()

    # Verificar valores nulos en cada columna
    valores_nulos = datos_limpios.isnull().sum()

    # Retornar DataFrame limpio y resumen de valores nulos
    return datos_limpios, valores_nulos


# Nombres de las columnas del conjunto de datos
columnas = [
    'duration', 'protocol_type', 'service', 'flag', 'src_bytes', 'dst_bytes', 'land', 'wrong_fragment',
    'urgent', 'hot', 'num_failed_logins', 'logged_in', 'num_compromised', 'root_shell', 'su_attempted',
    'num_root', 'num_file_creations', 'num_shells', 'num_access_files', 'num_outbound_cmds',
    'is_host_login', 'is_guest_login', 'count', 'srv_count', 'serror_rate', 'srv_serror_rate',
    'rerror_rate', 'srv_rerror_rate', 'same_srv_rate', 'diff_srv_rate', 'srv_diff_host_rate',
    'dst_host_count', 'dst_host_srv_count', 'dst_host_same_srv_rate', 'dst_host_diff_srv_rate',
    'dst_host_same_src_port_rate', 'dst_host_srv_diff_host_rate', 'dst_host_serror_rate',
    'dst_host_srv_serror_rate', 'dst_host_rerror_rate', 'dst_host_srv_rerror_rate', 'label'
]

# Ruta del archivo comprimido
ruta_archivo = 'kddcup.data.gz'

# Cargar y limpiar los datos
df_limpio, nulos = cargar_y_limpiar_datos(ruta_archivo, columnas)

# Mostrar los resultados
print(f"Forma de los datos después de la limpieza: {df_limpio.shape}")
print("Valores nulos por columna:")
print(nulos)
print("\nPrimeras filas de los datos limpios:")
print(df_limpio.head())



"""
los resultados se pueden interpretar como:
- el DataFrame tiene 1,074,992 filas y 42 columnas después de eliminar los duplicados. 
- no hay valores nulos en ninguna de las 42 columnas, todas las columnas tienen valores completos
- El DataFrame tiene más de un millón de registros y parece contener datos relacionados con conexiones 
    de red y etiquetas que indican si el tráfico es normal o anómalo.
"""