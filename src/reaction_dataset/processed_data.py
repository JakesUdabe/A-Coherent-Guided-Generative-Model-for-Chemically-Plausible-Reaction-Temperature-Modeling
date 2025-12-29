import os
import gzip
import json
import logging
from ord_schema.message_helpers import load_message
from ord_schema.proto import dataset_pb2
from google.protobuf.json_format import MessageToJson

# Configuración básica del logger
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")
logger = logging.getLogger("ord_json_mapper")

# --- CONFIGURACIÓN DE RUTAS ---
# 📌 ADVERTENCIA: Asegúrate de que BASE_DATA_DIR contenga tus subcarpetas (01, 02, etc.)
BASE_DATA_DIR = "/content/drive/MyDrive/CGGM/data/ord"
PROCESSED_DATA_DIR_NAME = "processed_data"  # Nombre de la carpeta a excluir
PROCESSED_DATA_DIR = os.path.join(BASE_DATA_DIR, PROCESSED_DATA_DIR_NAME)
# ------------------------------

def process_single_gz_to_jsonl(input_path: str, output_path: str):
    """
    Carga un dataset ORD (.gz), convierte cada reacción a JSONL y lo escribe.
    """
    
    # 1. Cargar el Dataset ORD (Protobuf)
    logger.info(f"Procesando: {os.path.basename(input_path)}")
    try:
        # load_message carga el archivo Protobuf (.gz) y lo deserializa
        dataset = load_message(input_path, dataset_pb2.Dataset)
    except Exception as e:
        # Este error ahora debería ser raro, a menos que el archivo .gz no sea un Protobuf válido
        logger.error(f"Error cargando Protobuf desde {input_path}. Detalle: {e}")
        return

    total_reactions = len(dataset.reactions)
    if total_reactions == 0:
        logger.warning(f"No se encontraron reacciones en {input_path}. Omitiendo.")
        return
    
    # 2. Procesar y Serializar a JSON Lines (JSONL)
    all_reactions_json_lines = []
    
    for rxn in dataset.reactions:
        # Convertimos el mensaje Protobuf de la reacción a una cadena JSON
        rxn_json_str = MessageToJson(
            message=rxn,
            # Configuración para JSON compacto y con nombres de campo de Protobuf
            including_default_value_fields=False,
            preserving_proto_field_name=True,
            indent=None, 
            sort_keys=False,
            use_integers_for_enums=False,
        )
        all_reactions_json_lines.append(rxn_json_str)

    # 3. Escribir el Archivo JSONL Comprimido
    logger.info(f"Escribiendo {total_reactions} reacciones a: {output_path}")
    
    # 'wt' = write text. gzip se encarga de la compresión.
    try:
        with gzip.open(output_path, 'wt', encoding='utf-8') as f:
            # Escribimos cada reacción como una línea JSON separada (JSON Lines)
            for line in all_reactions_json_lines:
                f.write(line + '\n')
        
        logger.info(f"✅ Conversión de {os.path.basename(input_path)} exitosa.")

    except Exception as e:
        logger.error(f"Error fatal al escribir el archivo {output_path}: {e}")


def recursive_ord_to_json_pipeline(base_input_dir: str, base_output_dir: str, extension: str = ".gz"):
    """
    Recorre recursivamente las subcarpetas de entrada, excluye la carpeta de salida, 
    y replica la estructura convirtiendo cada .gz a .jsonl.gz.
    """
    logger.info(f"Iniciando pipeline: De {base_input_dir} a {base_output_dir}")
    
    for root, dirs, files in os.walk(base_input_dir):
        
        # 📌 SOLUCIÓN CRÍTICA: Excluir el directorio de salida del recorrido
        if PROCESSED_DATA_DIR_NAME in dirs:
            logger.info(f"Excluyendo el directorio de salida del escaneo: {PROCESSED_DATA_DIR_NAME}")
            dirs.remove(PROCESSED_DATA_DIR_NAME)

        # La ruta relativa es la clave para mapear las subcarpetas (ej. '01', '02')
        relative_path = os.path.relpath(root, base_input_dir)
        output_sub_dir = os.path.join(base_output_dir, relative_path)
        
        # 1. Crear el subdirectorio de salida si no existe
        if not os.path.exists(output_sub_dir):
            os.makedirs(output_sub_dir, exist_ok=True)
            logger.debug(f"Directorio creado: {output_sub_dir}")

        for filename in files:
            # Solo procesamos los archivos .gz de la carpeta original
            if filename.lower().endswith(extension):
                input_path = os.path.join(root, filename)
                
                # 2. Definir el nombre del archivo de salida
                # Cambiar .gz a .jsonl.gz para indicar que es JSON Lines comprimido
                output_filename = filename.rsplit(extension, 1)[0] + ".jsonl.gz"
                output_path = os.path.join(output_sub_dir, output_filename)
                
                # 3. Llamar al procesador de archivo único
                process_single_gz_to_jsonl(input_path, output_path)

    logger.info("Pipeline de conversión completado.")


# --- EJECUCIÓN DEL SCRIPT ---
if __name__ == "__main__":
    recursive_ord_to_json_pipeline(BASE_DATA_DIR, PROCESSED_DATA_DIR)