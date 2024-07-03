"""
Nombre del código: Descarga de la base de datos KDD CUP con los datos de entrenamiento.

Alumno: Jimenez Poyatos, Pablo
Trabajo: Algoritmos de aprendizaje automático aplicados a problemas en ciberseguridad.

"""

import urllib.request
import gzip
import shutil

def descargar_url(image_url: str, filename: str) -> None:
    """
    Descarga un archivo desde una URL y lo guarda con el nombre especificado.

    Args:
        image_url (str): La URL desde donde se descargará el archivo.
        filename (str): El nombre con el cual se guardará el archivo descargado.

    Returns:
        None: La función no devuelve ningún valor, pero guarda el archivo descargado en el sistema de archivos local.
    """

    print(f"Downloading {filename}")
    try:
        with urllib.request.urlopen(image_url) as response:
            image_data = response.read()

        with open(filename, 'wb') as file:
            file.write(image_data)
        print(f"Documento descargado con éxito: {filename}")

    except:
        print(f"Ha fallado la descarga del documento: {filename}")


def descomprimir_gzip_file(gzip_file: str, output_file: str) -> None:
    """
    Descomprime un archivo .gz y lo guarda con el nombre especificado.

    Args:
        gzip_file (str): El nombre del archivo .gz a descomprimir.
        output_file (str): El nombre del archivo descomprimido.

    Returns:
        None: La función no devuelve ningún valor, pero guarda el archivo descomprimido en el sistema de archivos local.
    """

    with gzip.open(gzip_file, 'rb') as f_in:
        with open(output_file, 'wb') as f_out:
            shutil.copyfileobj(f_in, f_out)


if __name__ == "__main__":
    urls = ['http://kdd.ics.uci.edu/databases/kddcup99/kddcup.data.gz']

    for url in urls:
        filename = url.split("/")[-1]
        descargar_url(url, filename)
        archivo_gz = filename
        archivo_descomprimido = filename + '.txt'
        descomprimir_gzip_file(archivo_gz, archivo_descomprimido)

    print("¡Archivo descomprimido correctamente!")
