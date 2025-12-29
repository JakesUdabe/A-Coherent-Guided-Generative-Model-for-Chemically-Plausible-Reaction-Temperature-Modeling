import pandas as pd
import logging
import os

# Configuración de logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s [INFO] %(message)s")
logger = logging.getLogger(__name__)

# --- CONFIGURACIÓN DE RUTAS Y COLUMNAS ---
# Archivo de entrada
INPUT_FILE_PATH = "/content/drive/MyDrive/CGGM/data/ordord_parsed_reactions.csv.gz"

# Columnas clave requeridas para el análisis
KEY_COLUMNS = ['reactant_smiles_json', 'product_smiles_json', 'temperature_c']

# Archivo de salida (se guarda en el mismo directorio con un sufijo)
OUTPUT_FILE_PATH = os.path.join(
    os.path.dirname(INPUT_FILE_PATH), 
    "ord_filtered_key_reactions.csv.gz"
)
# ----------------------------------------

def process_and_filter_key_data(input_path: str, output_path: str, columns: list):
    """
    Carga el CSV, filtra filas con datos faltantes en las columnas clave,
    selecciona solo esas columnas, y exporta el resultado.
    """
    logger.info(f"Cargando archivo de entrada: {input_path}")
    
    try:
        # 1. Cargar solo las columnas clave para eficiencia.
        # Es necesario cargar todas las columnas si el CSV original tiene encabezados
        # y no queremos especificar usecols=None. Para el filtrado es más seguro.
        # Cargamos todas las columnas y luego filtramos para evitar problemas de índices.
        df = pd.read_csv(input_path)
        total_rows_before = len(df)
        logger.info(f"Archivo cargado. Filas iniciales: {total_rows_before}")

        # 2. Filtrar filas: Retener solo aquellas donde TODAS las columnas clave están llenas.
        # 'how=any' significa que si CUALQUIERA de las columnas especificadas es nula, se elimina la fila.
        df_filtered = df.dropna(subset=columns, how='any')
        total_rows_after = len(df_filtered)
        
        # 3. Seleccionar solo las columnas deseadas.
        # Las filas ya están filtradas, ahora reducimos el ancho (número de columnas).
        df_final = df_filtered[columns]
        
        # 4. Diagnóstico
        rows_removed = total_rows_before - total_rows_after
        logger.info("-" * 50)
        logger.info(f"Filas retenidas (Completas en {', '.join(columns)}): {total_rows_after}")
        logger.info(f"Filas eliminadas: {rows_removed}")
        logger.info(f"Porcentaje de retención: {round((total_rows_after / total_rows_before) * 100, 2)}%")
        logger.info("-" * 50)

        # 5. Exportar el DataFrame limpio
        logger.info(f"Exportando {len(df_final)} filas a: {output_path}")
        
        # 'compression='gzip' maneja la compresión, 'index=False' omite la columna de índice de pandas.
        df_final.to_csv(output_path, compression='gzip', index=False)
        
        logger.info("✅ Exportación completada. Archivo listo para análisis.")

    except FileNotFoundError:
        logger.error(f"El archivo no se encontró en la ruta: {input_path}")
    except KeyError as e:
        logger.error(f"Error: La columna {e} no se encontró en el archivo CSV. Verifica los nombres.")
    except Exception as e:
        logger.error(f"Ocurrió un error al procesar el archivo: {e}")

if __name__ == "__main__":
    process_and_filter_key_data(INPUT_FILE_PATH, OUTPUT_FILE_PATH, KEY_COLUMNS)