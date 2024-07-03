"""
Nombre del código: Modelo de autoencoder para resolver el problema de detección de intrusiones.

Alumno: Jimenez Poyatos, Pablo
Trabajo: Algoritmos de aprendizaje automático aplicados a problemas en ciberseguridad.

"""

import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.optimizers import Adam
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import CSVLogger, EarlyStopping
from data import generar_data
from confusionKDD_AE import confusion_matrix_pablo
from sklearn.metrics import precision_recall_curve
import matplotlib.pyplot as plt


def crear_ae(input_dim: int, encoding_dim1: int, encoding_dim2: int) -> Model:
    """
    Construye un modelo de autoencoder para detección de intrusiones.

    El autoencoder tiene una arquitectura de red neuronal con una capa de entrada, dos capas de codificación (encoder),
    dos capas de decodificación (decoder), y una capa de salida. La arquitectura se define por los hiperparámetros proporcionados.

    Args:
        input_dim (int): La dimensión de las entradas del autoencoder.
        encoding_dim1 (int): La dimensión de la primera capa de codificación.
        encoding_dim2 (int): La dimensión de la segunda capa de codificación.

    Returns:
        Model: El modelo de autoencoder construido con las capas definidas.
    """

    # Capa de entrada
    input_layer = Input(shape=(input_dim,))

    # Codificador
    encoded = Dense(encoding_dim1, activation='relu')(input_layer)
    encoded = Dense(encoding_dim2, activation='relu')(encoded)

    # Decodificador
    decoded = Dense(encoding_dim1, activation='relu')(encoded)
    output_layer = Dense(input_dim, activation='sigmoid')(decoded)

    # Modelo de autoencoder
    autoencoder = Model(inputs=input_layer, outputs=output_layer)

    autoencoder.summary()
    return autoencoder


def detect_anomalies(ae: Model, x_test: np.ndarray, threshold: float) -> np.ndarray:
    """
    Detecta anomalías en los datos de prueba utilizando el autoencoder.

    Se calcula el error de reconstrucción para cada instancia en el conjunto de prueba, y las instancias con un error
    mayor al umbral se etiquetan como anomalías.

    Args:
        ae (Model): El modelo de autoencoder entrenado.
        x_test (np.ndarray): Datos de prueba para detectar anomalías.
        threshold (float): Umbral para determinar si una instancia es una anomalía.

    Returns:
        np.ndarray: Un array de valores booleanos donde `True` indica una anomalía y `False` indica una instancia normal.
    """
    x_pred = ae.predict(x_test)
    reconstruction_errors = np.mean(np.abs(x_test - x_pred), axis=1)
    anomalies = reconstruction_errors > threshold
    return anomalies


def normales(X_train: np.ndarray, y_train: np.ndarray) -> (np.ndarray, np.ndarray):
    """
    Filtra las instancias normales del conjunto de entrenamiento.

    Se extraen las instancias de entrenamiento que se etiquetan como normales (es decir, con `y_train` igual a 1).

    Args:
        X_train (np.ndarray): Conjunto de datos de entrenamiento.
        y_train (np.ndarray): Etiquetas de las instancias de entrenamiento.

    Returns:
        (np.ndarray, np.ndarray): Un tuple con dos arrays: `normales` contiene las instancias normales y `normales_y` contiene sus etiquetas.
    """
    normales = []
    normales_y = []
    for i in range(X_train.shape[0]):
        if int(y_train[i][0]) == 1:
            normales.append(X_train[i])
            normales_y.append(y_train[i])
    return normales, normales_y


def crear_etiquetas_binarias(y_data: np.ndarray) -> list:
    """
    Crea etiquetas binarias para la evaluación del modelo basadas en las etiquetas originales.

    Esta función convierte las etiquetas originales en un formato binario donde 0 representa una normalidad y 1 representa malware.

    Args:
        y_data (np.ndarray): Etiquetas originales de clase.

    Returns:
        list: Lista de etiquetas binarias, 0 para normales y 1 para anomalías.
    """
    etiquetas_verdaderas = []
    for y in y_data:
        if y[0] == 1:
            etiquetas_verdaderas.append(0)
        else:
            etiquetas_verdaderas.append(1)
    return etiquetas_verdaderas


def guardar_imagen_umbral(thresholds: np.ndarray, precisions: np.ndarray, recalls: np.ndarray) -> None:
    """
    Guarda un gráfico de la precisión y el recall en función del umbral.

    La función genera una gráfica de precisión y recall en función del umbral de error de reconstrucción del autoencoder.

    Args:
        thresholds (np.ndarray): Array de umbrales utilizados para la evaluación del modelo.
        precisions (np.ndarray): Array de precisiones correspondientes a cada umbral.
        recalls (np.ndarray): Array de recalls correspondientes a cada umbral.

    Returns:
        None: La función guarda la imagen en un archivo llamado `threshold.png`.
    """
    plt.figure(figsize=(8, 6))
    plt.plot(thresholds, precisions[:-1], "b--", label="Precision")
    plt.plot(thresholds, recalls[:-1], "g-", label="Recall")
    plt.xlabel("Threshold")
    plt.ylabel("Precision/Recall")
    plt.title("Precision-Recall vs Threshold")
    plt.legend(loc="best")
    plt.grid()
    plt.savefig("threshold.png")


if __name__ == "__main__":
    # Hiperparámetros iniciales
    file_name = "KDD_AE"
    input_dim = 122
    encoding_dim1 = 32
    encoding_dim2 = 5
    batch_size = 32
    epochs = 30
    learning_rate = 0.001


    # Preprocesamiento de los datos
    data, labels, _ = generar_data()
    data = data.reshape((data.shape[0], data.shape[1])).astype(np.float32)
    X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=0.15, random_state=0)

    normal, normalesy = normales(X_train, y_train)
    X_train_normal = np.array(normal)
    y_train_normales = np.array(normalesy)


    # Construir y entrenar el AE
    csv_logger = CSVLogger('KDD_AE.csv', append=False)
    early_stopping = EarlyStopping(monitor='val_loss', patience=5,
                                   restore_best_weights=True)

    ae = crear_ae(input_dim, encoding_dim1, encoding_dim2)
    ae.compile(optimizer=Adam(learning_rate=learning_rate),
               loss='mean_absolute_error')
    ae.fit(X_train_normal, X_train_normal, epochs=epochs, batch_size=batch_size, validation_split=0.1,
           callbacks=[csv_logger, early_stopping], shuffle=True)

    ae.save(file_name + '.keras')


    # Calcular el umbral para decidir si una instancia es malware o no.
    X_train_pred = ae.predict(X_train)  # Obtener las predicciones del modelo en el conjunto de entrenamiento
    train_reconstruction_errors = np.mean(np.abs(X_train - X_train_pred),
                                          axis=1)  # Calcular el error de reconstrucción para cada instancia
    etiquetas_verdaderas_train = crear_etiquetas_binarias(
        y_train)  # Crear etiquetas binarias para el conjunto de entrenamiento

    precisions, recalls, thresholds = precision_recall_curve(etiquetas_verdaderas_train,
                                                             train_reconstruction_errors)  # Calcular precisión y recall para diferentes umbrales
    guardar_imagen_umbral(thresholds, precisions, recalls)  # Guardar el gráfico de precisión y recall vs umbral
    umbral = 0.02  # Umbral seleccionado para detectar anomalías basado en el análisis del gráfico


    # Evaluar el modelo
    anomalias = detect_anomalies(ae, X_test, umbral)  # Detectar anomalías en el conjunto de prueba
    etiquetas_verdaderas = crear_etiquetas_binarias(y_test)  # Crear etiquetas binarias para el conjunto de prueba

    confusion_matrix_pablo(file_name, etiquetas_verdaderas, anomalias)  # Generar y guardar las matrices de confusión




