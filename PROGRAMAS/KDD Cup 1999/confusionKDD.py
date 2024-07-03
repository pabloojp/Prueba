"""
Nombre del código: Evaluación y visualización de un modelo de clasificación.

Alumno: Jimenez Poyatos, Pablo
Trabajo: Algoritmos de aprendizaje automático aplicados a problemas en ciberseguridad.

"""

from sklearn.metrics import confusion_matrix
from keras.models import load_model

import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

def convertir_a_2_dimensiones(vectores: np.ndarray) -> np.ndarray:
    """
    Convierte un vector de clases a una representación binaria (0 o 1).

    Args:
        vectores (np.ndarray): Array de etiquetas de clase. Cada valor es la clase predicha o verdadera.

    Returns:
        np.ndarray: Array con valores binarios (0 o 1). 0 para la clase 0, 1 para la clase 1.
    """
    nuevos_vectores = []
    for vector in vectores:
        if vector == 0:
            nuevos_vectores.append(0)
        else:
            nuevos_vectores.append(1)
    return np.array(nuevos_vectores)


def confusion_matrix_pablo(nombre: str, X_test: np.ndarray, y_test: np.ndarray, guardar: str) -> None:
    """
    Carga un modelo entrenado, realiza predicciones sobre un conjunto de prueba, y genera matrices de confusión.

    Args:
        nombre (str): Nombre del archivo del modelo guardado que se va a cargar.
        X_test (np.ndarray): Datos de entrada del conjunto de prueba.
        y_test (np.ndarray): Etiquetas verdaderas del conjunto de prueba en formato one-hot.
        guardar (str): Ruta base del archivo donde se guardarán las imágenes de las matrices de confusión.

    Returns:
        None: La función no devuelve ningún valor, pero guarda las imágenes de las matrices de confusión.
    """
    # Cargar el modelo desde el archivo
    model = load_model(nombre)

    # Hacer predicciones con el modelo
    y_pred = model.predict(X_test)
    # Convertir las predicciones a clases
    y_pred_clases = np.argmax(y_pred, axis=-1)
    y_pred_clases = convertir_a_2_dimensiones(y_pred_clases)

    # Convertir las etiquetas verdaderas a clases
    y_true_clases = np.argmax(y_test, axis=-1)
    y_true_clases = convertir_a_2_dimensiones(y_true_clases)

    # Matriz de confusión con el número de muestras clasificadas bien o mal.
    conf_matrix = confusion_matrix(y_true_clases, y_pred_clases)
    plt.figure(figsize=(10, 8))
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues',
                xticklabels=["Normal", "Anomalía"],
                yticklabels=["Normal", "Anomalía"],
                annot_kws={"fontsize": 12})  # Tamaño de fuente de las anotaciones
    plt.xlabel('Predicho')
    plt.ylabel('Real')
    plt.title('Matriz de Confusión')
    plt.savefig(guardar + '.png')
    plt.show()

    # Matriz de confusión con porcentajes
    conf_matrix_norm = confusion_matrix(y_true_clases, y_pred_clases, normalize='true')
    plt.figure(figsize=(10, 8))
    sns.heatmap(conf_matrix_norm, annot=True, fmt='.2f', cmap='Blues',
                xticklabels=["Normal", "Anomalía"],
                yticklabels=["Normal", "Anomalía"],
                annot_kws={"fontsize": 12})
    plt.xlabel('Predicho')
    plt.ylabel('Real')
    plt.title('Matriz de Confusión Normalizada')
    plt.savefig(guardar + 'Norm1.png')
    plt.show()
