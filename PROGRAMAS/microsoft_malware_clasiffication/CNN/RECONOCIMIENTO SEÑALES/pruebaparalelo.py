# -*- coding: utf-8 -*-
"""
Created on Thu Feb  1 12:33:13 2024

@author: pjime
"""
import numpy as np                                     # Importamos la biblioteca NumPy para operaciones num√©ricas eficientes                 
import os                                              # Importamos el m√≥dulo os para interactuar con el sistema operativo
from PIL import Image                                  # Importamos la clase Image del m√≥dulo PIL para trabajar con im√°genes
import multiprocessing as mp
from time import perf_counter



def reparte(numero: int, nr_partes: int) -> list[int]:
    """Divide `numero` en `nr_partes` partes enteras más o menos iguales.
    Por ejemplo, reparte(10, 3) devuelve la lista [4, 3, 3]."""
    cociente, resto = divmod(numero, nr_partes)
    return [cociente + 1] * resto + [cociente] * (nr_partes - resto)


def generar_listas(N, LISTA):
    resultado = []
    inicio = 0

    for longitud in LISTA:
        nueva_lista = list(range(inicio, inicio + longitud))
        resultado.append(nueva_lista)
        inicio += longitud

    return resultado

def concat(xss):
    xs = []
    for i in xss:
        xs += i
    return xs

def first(par):
    return par[0]

def second(par):
    return par[1]

def load_and_preprocess_data(clas_min, clas_max):
    data = []
    label = []
    for i in range(clas_min,clas_max+1):
        ruta = os.path.join('Train', str(i))  # Construimos la ruta para cada clase
        images = os.listdir(ruta)                             # Listamos todas las im√°genes en la carpeta de la clase

        # Recorremos cada imagen en la clase
        for a in images:
            try :
                image0 = os.path.join(ruta,a)
                image1 = Image.open(image0)   # Cargamos la imagen utilizando PIL
                image2 = image1.resize((30, 30))       # Redimensionamos la imagen a 30x30 p√≠xeles
                image3 = np.array(image2)              # Convertimos la imagen a un array de NumPy
                data.append(image3)                   # Agregamos la imagen a la lista de datos
                label.append(i)                     # Agregamos la etiqueta correspondiente a la lista de etiquetas
            except:
                pass

    return data, label


def load_and_preprocess_data_paralelo(nr_clases, nr_procesos):
    p = mp.Pool(nr_procesos)
    lista = reparte(nr_clases, nr_procesos)
    clases_proces = generar_listas(nr_clases, lista)
    args_list = [(clases_proces[k][0],clases_proces[k][-1]) for k in range(len(clases_proces))]
    resultados = p.starmap(load_and_preprocess_data, args_list)
    data = concat(list(map(first,resultados)))
    label = concat(list(map(second,resultados)))
    data = np.array(data)
    label = np.array(label)
    return data, label

if __name__ == "__main__":
    for i in range(1,9):
        t1 = perf_counter()
        data, labels = load_and_preprocess_data_paralelo(43, i)
        t2 = perf_counter()
        print(f'Tiempo con {i} procesadores: {t2-t1}')
