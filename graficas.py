import os
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime

def comparar_graficas(csv_files, save_dir="graficas"):
    """
    Compara múltiples archivos CSV generando gráficas para métricas específicas.

    Args:
        csv_files (list): Lista de rutas a los archivos CSV.
        save_dir (str): Carpeta base donde se guardarán las gráficas.
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_folder = os.path.join(save_dir, f"comparacion_{timestamp}")
    os.makedirs(output_folder, exist_ok=True)

    metrics = ["train_loss", "val_loss", "val_accuracy", "val_precision", "val_recall", "val_mIoU"]

    dataframes = {}
    for csv_file in csv_files:
        try:
            df = pd.read_csv(csv_file, delimiter=",")  # Cambiado a ',' como separador
            df.columns = df.columns.str.strip()  # Eliminar espacios adicionales

            # Verificación de columnas
            print(f"Columnas en {csv_file}: {df.columns.tolist()}")

            if 'epoch' not in df.columns:
                raise KeyError(f"Error: La columna 'epoch' no se encontró en {csv_file}")

            df['epoch'] = df['epoch'].astype(int)  # Asegurar tipo de dato entero en epoch
            
            file_name = os.path.basename(csv_file).replace('.csv', '')
            dataframes[file_name] = df
            print(f"Archivo '{csv_file}' cargado correctamente.")

        except Exception as e:
            print(f"Error al procesar {csv_file}: {e}")
            continue

    # Generar gráficas para cada métrica
    for metric in metrics:
        plt.figure(figsize=(10, 6))
        
        for file_name, df in dataframes.items():
            if metric in df.columns:
                plt.plot(df['epoch'], df[metric], label=file_name)
            else:
                print(f"Advertencia: '{metric}' no encontrado en {file_name}")

        plt.xlabel('Epoch')
        plt.ylabel(metric.replace('_', ' ').title())
        plt.title(f'Comparison of {metric} with 13 features')

        plt.legend()
        plt.grid(True)

        output_path = os.path.join(output_folder, f"{metric}.png")
        plt.savefig(output_path)
        plt.close()

    print(f"Gráficas guardadas en: {output_folder}")

# Lista de archivos CSV a comparar
csv_files = [
    "ALLPCA/GAT.csv",
    "ALLPCA/GCN.csv",
    "ALLPCA/Unet.csv",
    "ALLPCA/Unet2.csv",
    "ALLPCA/EdgeGAT.csv"
]

comparar_graficas(csv_files)
