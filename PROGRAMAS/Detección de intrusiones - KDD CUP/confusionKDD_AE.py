"""
Nombre del código: Evaluación de un modelo de clasificación mediante la matriz de confusión.

Alumno: Jimenez Poyatos, Pablo
Trabajo: Algoritmos de aprendizaje automático aplicados a problemas en ciberseguridad.

"""

import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

from sklearn.metrics import confusion_matrix


def confusion_matrix_pablo(nombre: str, y_true_classes: np.ndarray, y_pred_classes: np.ndarray) -> None:
    """
    Genera y guarda dos matrices de confusión basadas en las etiquetas verdaderas y predicciones de un modelo.

    Esta función crea dos matrices de confusión:
    1. La primera matriz muestra el número de muestras clasificadas correctamente y erróneamente.
    2. La segunda matriz muestra los porcentajes de muestras clasificadas correctamente y erróneamente.

    Ambas matrices se guardan como imágenes en el archivo especificado por `nombre`.

    Args:
        nombre (str): El nombre base del archivo donde se guardarán las imágenes de las matrices de confusión.
        y_true_classes (np.ndarray): Etiquetas verdaderas de clase para el conjunto de datos. Se espera que sea un array con valores de clase binaria (0 o 1).
        y_pred_classes (np.ndarray): Etiquetas predichas por el modelo para el conjunto de datos. Se espera que sea un array con valores de clase binaria (0 o 1).

    Returns:
        None: La función no devuelve ningún valor, pero guarda las imágenes de las matrices de confusión.
    """
    # Crea una matriz de confusión con el número de muestras clasificadas bien o mal.
    conf_matrix = confusion_matrix(y_true_classes, y_pred_classes)
    plt.figure(figsize=(10, 8))
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues',
                xticklabels=["Normal", "Anomalía"],
                yticklabels=["Normal", "Anomalía"],
                annot_kws={"fontsize": 12})  # Tamaño de fuente de las anotaciones
    plt.xlabel('Predicho')
    plt.ylabel('Real')
    plt.title('Matriz de Confusión')
    plt.savefig(nombre + '.png')
    plt.show()

    # Crea una matriz de confusión normalizada que muestra los porcentajes de muestras clasificadas correctamente y erróneamente.
    conf_matrix_norm = confusion_matrix(y_true_classes, y_pred_classes, normalize='true')
    plt.figure(figsize=(10, 8))
    sns.heatmap(conf_matrix_norm, annot=True, fmt='.2f', cmap='Blues',
                xticklabels=["Normal", "Anomalía"],
                yticklabels=["Normal", "Anomalía"],
                annot_kws={"fontsize": 12})
    plt.xlabel('Predicho')
    plt.ylabel('Real')
    plt.title('Matriz de Confusión Normalizada')
    plt.savefig(nombre + 'Norm.png')
    plt.show()
