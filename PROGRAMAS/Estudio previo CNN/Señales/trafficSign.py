"""
Nombre del código: Modelo de CNN de reconocimiento de señales.
Guiado por: Tutorial de Kaggle (acceso al enlace el 12 de noviembre)
        https://www.kaggle.com/code/yacharki/traffic-signs-image-classification-96-cnn#6.-Training-the-Model
Alumno: Jiménez Poyatos, Pablo


Además, he tenido que descargarme las imágenes de entrenamiento en diferentes carpetas (cada clase en una carpeta)
y guardarlas en la misma carpeta donde estaba este script.
"""

# Importación de bibliotecas
import numpy as np                                     # Operaciones numéricas eficientes
import pandas as pd                                    # Manipulación y análisis de datos
import tensorflow as tf                                # Biblioteca para aprendizaje automático
import os                                              # Interacción con el sistema operativo
from PIL import Image                                  # Trabajar con imágenes
from sklearn.model_selection import train_test_split   # División de datos en conjuntos de entrenamiento y prueba
from keras.utils import to_categorical                 # Codificación de etiquetas en formato one-hot
from keras.models import Sequential, load_model        # Modelo lineal para apilar capas de red neuronal
from keras.layers import Conv2D, MaxPool2D, Dense, Flatten, Dropout  # Construcción de una red neuronal convolucional
import matplotlib.pyplot as plt                        # Visualización de datos

def load_and_preprocess_data():
    """
    Carga y preprocesa los datos de entrenamiento desde carpetas.

    Devuelve:
    - data: Array de imágenes de entrenamiento.
    - labels: Array de etiquetas correspondientes.
    """
    data = []                                         # Lista para almacenar datos de imágenes
    labels = []                                       # Lista para almacenar etiquetas de imágenes
    classes = 43                                      # Número total de clases en el conjunto de datos

    for i in range(classes):
        ruta = os.path.join('Train', str(i))  # Ruta para cada clase
        images = os.listdir(ruta)             # Lista de todas las imágenes en la carpeta de la clase

        for a in images:
            try:
                image_path = os.path.join(ruta, a)
                image = Image.open(image_path)        # Cargar la imagen
                image = image.resize((30, 30))        # Redimensionar a 30x30 píxeles
                image_array = np.array(image)         # Convertir a array de NumPy
                data.append(image_array)              # Agregar a la lista de datos
                labels.append(i)                      # Agregar la etiqueta correspondiente
            except:
                pass

    data = np.array(data)      # Convertir lista de datos a array de NumPy
    labels = np.array(labels)  # Convertir lista de etiquetas a array de NumPy
    return data, labels

def split_and_encode_data(data, labels, test_size=0.2, random_state=42):
    """
    Divide los datos en conjuntos de entrenamiento y prueba, y codifica las etiquetas.

    Argumentos:
    - data: Array de imágenes.
    - labels: Array de etiquetas.
    - test_size: Proporción del conjunto de prueba.
    - random_state: Semilla para la división aleatoria.

    Devuelve:
    - X_train: Conjunto de datos de entrenamiento.
    - X_test: Conjunto de datos de prueba.
    - y_train: Etiquetas de entrenamiento codificadas.
    - y_test: Etiquetas de prueba codificadas.
    """
    X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=test_size, random_state=random_state)
    y_train = to_categorical(y_train, 43)  # Codificar etiquetas en one-hot
    y_test = to_categorical(y_test, 43)    # Codificar etiquetas en one-hot
    return X_train, X_test, y_train, y_test

def build_model(input_shape):
    """
    Construye y compila el modelo CNN.

    Argumentos:
    - input_shape: Dimensiones de entrada del modelo.

    Devuelve:
    - model: Modelo CNN compilado.
    """
    model = Sequential()                                                                           # Modelo secuencial

    model.add(Conv2D(filters=32, kernel_size=(5, 5), activation='relu', input_shape=input_shape))  # Capa de convolución
    model.add(Conv2D(filters=32, kernel_size=(5, 5), activation='relu'))                           # Capa de convolución
    model.add(MaxPool2D(pool_size=(2, 2)))                                                         # Capa de agrupación
    model.add(Dropout(rate=0.25))                                                                  # Capa de Dropout

    model.add(Conv2D(filters=64, kernel_size=(3, 3), activation='relu'))                           # Capa de convolución
    model.add(Conv2D(filters=64, kernel_size=(3, 3), activation='relu'))                           # Capa de convolución
    model.add(MaxPool2D(pool_size=(2, 2)))                                                         # Capa de agrupación
    model.add(Dropout(rate=0.25))                                                                  # Capa de Dropout

    model.add(Flatten())                                                                           # Aplanar las características
    model.add(Dense(256, activation='relu'))                                                       # Capa completamente conectada
    model.add(Dropout(rate=0.5))                                                                   # Capa de Dropout
    model.add(Dense(43, activation='softmax'))                                                     # Capa de salida

    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])         # Compilación del modelo

    model.summary()  # Imprimir resumen de la arquitectura
    return model

def train_model(model, file_name, X_train, y_train, X_test, y_test, batch_size=32, epochs=1):
    """
    Entrena el modelo y lo guarda en un archivo.

    Argumentos:
    - model: Modelo CNN a entrenar.
    - file_name: Nombre del archivo donde se guardará el modelo.
    - X_train: Conjunto de datos de entrenamiento.
    - y_train: Etiquetas de entrenamiento.
    - X_test: Conjunto de datos de prueba.
    - y_test: Etiquetas de prueba.
    - batch_size: Tamaño del lote para el entrenamiento.
    - epochs: Número de épocas de entrenamiento.

    Devuelve:
    - None
    """
    with tf.device('/GPU:0'):   # Utilizar GPU si está disponible
        model.fit(X_train, y_train, batch_size=batch_size, epochs=epochs, validation_data=(X_test, y_test))
    model.save(file_name)       # Guardar el modelo entrenado

if __name__ == "__main__":
    # Cargar y preprocesar los datos
    data, labels = load_and_preprocess_data()

    # Visualizar la cantidad de datos por etiqueta
    label_counts = np.bincount(labels)
    plt.bar(range(len(label_counts)), label_counts, tick_label=range(len(label_counts)))
    for i, count in enumerate(label_counts):
        plt.text(i, count + 0.1, str(count), ha='center', va='bottom')
    plt.xlabel('Etiquetas')
    plt.ylabel('Cantidad de Datos')
    plt.title('Cantidad de Datos por Etiqueta')
    plt.show()

    # Dividir y codificar los datos
    X_train, X_test, y_train, y_test = split_and_encode_data(data, labels)

    # Construir el modelo
    model = build_model(X_train.shape[1:])

    # Entrenar el modelo y guardarlo en un archivo
    train_model(model, 'traffic_classifier.keras', X_train, y_train, X_test, y_test, batch_size=32, epochs=5)







'''
"""
Para predecir la señal que aparece en una imagen, tenemos que cargar el modelo despues de 20 epochs y
la imagen. Esta última tenemos que pasarla a una imagen con tres canales  y redimensionarla a 30 x 30 pixeles.
Despues de que mi modelo prediga cual sería su etiqueta, le asignamos su nombre correspondiente del diccionario 
clases.
"""
#predecir('traffic_classifier.keras', 'prohib.jpg')
def predecir(model, nombre):
    os.getcwd()
    # Cargar el modelo previamente entrenado
    model = load_model(model)
    
    # Cargar la imagen
    image_path = nombre 
    image0 = Image.open(image_path)
    image1 = image0.resize((30, 30))
    image2 = image1.convert('RGB')
    image3 = np.array(image2)
    # Normalizar los valores de píxeles
    
    # Realizar la predicción con el modelo cargado
    prediction = model.predict(np.array([image3]))  # Asegurarse de que sea un arreglo de forma (1, 30, 30, 3)
    
    # Obtener la etiqueta predicha (índice de la clase con mayor probabilidad)
    predicted_class = np.argmax(prediction)
    
    # Crear un diccionario que mapea las clases a sus etiquetas
    clases = { 
        0: 'Límite de velocidad (20 km/h)',
        1: 'Límite de velocidad (30 km/h)', 
        2: 'Límite de velocidad (50 km/h)', 
        3: 'Límite de velocidad (60 km/h)', 
        4: 'Límite de velocidad (70 km/h)', 
        5: 'Límite de velocidad (80 km/h)', 
        6: 'Fin del límite de velocidad (80 km/h)', 
        7: 'Límite de velocidad (100 km/h)', 
        8: 'Límite de velocidad (120 km/h)', 
        9: 'Prohibido adelantar', 
        10: 'Prohibido adelantar vehículos de más de 3.5 toneladas', 
        11: 'Derecho de paso en intersección', 
        12: 'Carretera con prioridad', 
        13: 'Ceder el paso', 
        14: 'Detenerse', 
        15: 'Prohibido el paso de vehículos', 
        16: 'Prohibido el paso de vehículos de más de 3.5 toneladas',
        17: 'Prohibido el acceso', 
        18: 'Precaución general', 
        19: 'Curva peligrosa a la izquierda', 
        20: 'Curva peligrosa a la derecha', 
        21: 'Curva doble', 
        22: 'Carretera con baches', 
        23: 'Carretera resbaladiza', 
        24: 'Carretera se estrecha a la derecha', 
        25: 'Trabajo en la carretera', 
        26: 'Señales de tráfico', 
        27: 'Peatones', 
        28: 'Cruce de niños', 
        29: 'Cruce de bicicletas', 
        30: 'Precaución: hielo/nieve',
        31: 'Cruce de animales salvajes', 
        32: 'Fin de límites de velocidad y adelantamiento', 
        33: 'Girar a la derecha', 
        34: 'Girar a la izquierda', 
        35: 'Solo adelante', 
        36: 'Ir recto o girar a la derecha', 
        37: 'Ir recto o girar a la izquierda', 
        38: 'Mantenerse a la derecha', 
        39: 'Mantenerse a la izquierda', 
        40: 'Circulación obligatoria en rotonda', 
        41: 'Fin de la prohibición de adelantar', 
        42: 'Fin de la prohibición de adelantar vehículos de más de 3.5 toneladas'
    }
    
    # Obtener la etiqueta correspondiente a la clase predicha
    predicted_label = clases[predicted_class]
    
    probabilities = prediction[0]
    
    # Imprimir la etiqueta predicha
    print(f'Clase predicha: {predicted_label}')
    
    print('Probabilidades:')
    for i, prob in enumerate(probabilities):
        print(f'{clases[i]}: {prob:.4f}')

'''