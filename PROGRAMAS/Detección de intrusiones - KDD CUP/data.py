"""
Nombre del código: Procesar los datos KDD CUP 1999.

Alumno: Jimenez Poyatos, Pablo
Trabajo: Algoritmos de aprendizaje automático aplicados a problemas en ciberseguridad.

"""

import numpy as np
from collections import defaultdict

# VARIABLES GLOBALES

# Mapeo de los tipos de malware con su clase correspondiente
class_mapping = {
    'normal.': 0, 'pod.': 1, 'land.': 1, 'neptune.': 1, 'smurf.': 1, 'back.': 1, 'teardrop.': 1,
    'warezclient.': 2, 'phf.': 2, 'imap.': 2, 'spy.': 2, 'ftp_write.': 2, 'guess_passwd.': 2,
    'warezmaster.': 2, 'multihop.': 2, 'rootkit.': 3, 'loadmodule.': 3, 'buffer_overflow.': 3,
    'perl.': 3, 'portsweep.': 4, 'ipsweep.': 4, 'satan.': 4, 'nmap.': 4
}


# FUNCIONES

def concat(lista: list) -> list:
    """
    Concatena los elementos de una lista que son listas.

    Args:
        lista (list): Lista que puede contener listas.

    Returns:
        list: Lista con los elementos concatenados.
    """
    resultado = []
    for elemento in lista:
        if isinstance(elemento, list):
            resultado.extend(elemento)
        else:
            resultado.append(elemento)
    return resultado


def crear_diccionario_one_hot(elementos_set: dict) -> dict:
    """
    Crea un diccionario de codificación one-hot para los elementos en un conjunto.

    Args:
        elementos_set (dict): Diccionario de elementos a codificar.

    Returns:
        dict: Diccionario con las claves originales y sus correspondientes vectores one-hot.
    """
    elementos_list = sorted(elementos_set.keys())
    diccionario_one_hot = {}
    for idx, elemento in enumerate(elementos_list):
        vector_one_hot = np.zeros(len(elementos_list), dtype=int)
        vector_one_hot[idx] = 1
        diccionario_one_hot[elemento] = vector_one_hot.tolist()
    return diccionario_one_hot


def crear_diccionario_one_hotL() -> tuple:
    """
    Crea un diccionario de codificación one-hot para las etiquetas de clase.

    Returns:
        tuple: Diccionario de codificación one-hot y diccionario de etiquetas.
    """
    normal = dict(zip(['normal.'], [0]))
    dos = dict(zip(['pod.', 'land.', 'neptune.', 'smurf.', 'back.', 'teardrop.'], [1] * 6))
    r2l = dict(
        zip(['warezclient.', 'phf.', 'imap.', 'spy.', 'ftp_write.', 'guess_passwd.', 'warezmaster.', 'multihop.'],
            [2] * 8))
    u2r = dict(zip(['rootkit.', 'loadmodule.', 'buffer_overflow.', 'perl.'], [3] * 4))
    probing = dict(zip(['portsweep.', 'ipsweep.', 'satan.', 'nmap.'], [4] * 4))
    elementos_dict = normal | dos | r2l | u2r | probing

    clases = set(class_mapping.values())
    diccionario_one_hot = {}
    for etiqueta in elementos_dict.keys():
        clase = class_mapping.get(etiqueta)
        vector_one_hot = np.zeros(len(clases), dtype=int)
        if clase is not None:
            vector_one_hot[clase] = 1
        diccionario_one_hot[etiqueta] = vector_one_hot.tolist()
    return diccionario_one_hot, elementos_dict


def actualizar_min_max(elementos: list, minim: list, maxim: list) -> None:
    """
    Actualiza los valores mínimos y máximos de los elementos.

    Args:
        elementos (list): Lista de elementos a evaluar.
        minim (list): Lista de valores mínimos por posición.
        maxim (list): Lista de valores máximos por posición.

    Returns:
        None: La función no devuelve ningún valor.
    """
    for i in range(len(elementos) - 1):
        if i not in [1, 2, 3]:
            valor = float(elementos[i])
            if valor < minim[i]:
                minim[i] = valor
            if valor > maxim[i]:
                maxim[i] = valor


def generar_diccionario_onehot(file_path: str) -> tuple:
    """
    Genera diccionarios de codificación one-hot para los elementos en un archivo.

    Args:
        file_path (str): Ruta al archivo de datos.

    Returns:
        tuple: Diccionarios de codificación one-hot y listas de valores máximos y mínimos.
    """
    elementos_set1 = defaultdict(int)
    elementos_set2 = defaultdict(int)
    elementos_set3 = defaultdict(int)

    maxim = [0] * 41
    minim = [235] * 41

    with open(file_path, 'r') as archivo:
        for linea in archivo:
            elementos = linea.strip().split(',')
            if len(elementos) >= 4:
                elementos_set1[elementos[1]] += 1
                elementos_set2[elementos[2]] += 1
                elementos_set3[elementos[3]] += 1

    diccionario_one_hot1 = crear_diccionario_one_hot(elementos_set1)
    diccionario_one_hot2 = crear_diccionario_one_hot(elementos_set2)
    diccionario_one_hot3 = crear_diccionario_one_hot(elementos_set3)

    return diccionario_one_hot1, diccionario_one_hot2, diccionario_one_hot3, maxim, minim


def crear_data(diccionario1: dict, diccionario2: dict, diccionario3: dict, maxim: list, minim: list, file_path: str,
               labels_one_hot: dict) -> tuple:
    """
    Crea los datos normalizados y transformados a partir del archivo de datos.

    Args:
        diccionario1 (dict): Diccionario one-hot para el segundo elemento.
        diccionario2 (dict): Diccionario one-hot para el tercer elemento.
        diccionario3 (dict): Diccionario one-hot para el cuarto elemento.
        maxim (list): Lista de valores máximos para normalización.
        minim (list): Lista de valores mínimos para normalización.
        file_path (str): Ruta al archivo de datos.
        labels_one_hot (dict): Diccionario de etiquetas con codificación one-hot.

    Returns:
        tuple: Arrays de datos normalizados, etiquetas y "imágenes" de los datos.
    """
    data = []
    image = []
    labels = []

    with open(file_path, 'r') as archivo:
        lineas = archivo.readlines()
        for i, linea in enumerate(lineas):
            elementos = linea.split(',')
            etiqueta = elementos[-1][:-1]
            labels.append(labels_one_hot[etiqueta])

            elementos[1] = diccionario1[elementos[1]]
            elementos[2] = diccionario2[elementos[2]]
            elementos[3] = diccionario3[elementos[3]]
            for j in range(len(elementos) - 1):
                if j not in [1, 2, 3] and maxim[j] != minim[j]:
                    elementos[j] = (float(elementos[j]) - minim[j]) / (maxim[j] - minim[j])

            datos = concat(elementos[:-1])
            data.append(datos)

            datos_image = datos.copy()
            datos_image.pop(100)
            datos_image = np.array(datos_image)
            image_data = datos_image.reshape((11, 11))[..., np.newaxis]
            image.append(image_data)

            if (i + 1) % 100000 == 0:
                print(f'Procesadas {i + 1} líneas')

    image = np.array(image, dtype=np.float32)
    data = np.array(data, dtype=np.float32)
    labels = np.array(labels).astype(np.int64)
    return data, labels, image


def generar_data() -> tuple:
    """
    Genera los datos procesados a partir del archivo KDD CUP.

    Returns:
        tuple: Arrays de datos normalizados, etiquetas y "imágenes" de los datos.
    """
    labels_one_hot, _ = crear_diccionario_one_hotL()

    file_path = 'kddcup.data.gz.txt'
    diccionario1, diccionario2, diccionario3, maxim, minim = generar_diccionario_onehot(file_path)

    maxim = [58329.0, 0, 0, 0, 1379963888.0, 1309937401.0, 1.0, 3.0, 14.0, 77.0, 5.0, 1.0, 7479.0, 1.0, 2.0, 7468.0,
             43.0, 2.0, 9.0, 0, 1.0, 1.0, 511.0, 511.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 255.0, 255.0, 1.0, 1.0, 1.0,
             1.0, 1.0, 1.0, 1.0, 1.0]
    minim = [0.0, 235, 235, 235, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
             0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]

    resultados = crear_data(diccionario1, diccionario2, diccionario3, maxim, minim, file_path, labels_one_hot)
    return resultados
