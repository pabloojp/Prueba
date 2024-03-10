# -*- coding: utf-8 -*-
"""
Created on Thu Feb  1 12:48:36 2024

@author: pjime
"""

import numpy as np

# Lista de pares (data, label)
lista_pares = [(np.array([1, 2]), 0), (np.array([3, 4]), 1), (np.array([5, 6]), 2)]

# Desempaquetar y concatenar datos y etiquetas
todos_datos = [dato for dato, _ in lista_pares]
todos_labels = [label for _, label in lista_pares]

# Concatenar todas las listas de datos y etiquetas
datos_concatenados = np.concatenate(todos_datos)
labels_concatenados = np.array(todos_labels)

# Mostrar resultados
print(f"Todos los datos concatenados:{ datos_concatenados}")
print(f"Todos los labels concatenados:{ labels_concatenados}")
