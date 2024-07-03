"""
Nombre del código: Modelo de CNN para resolver el problema de detección de intrusiones.

Alumno: Jimenez Poyatos, Pablo
Trabajo: Algoritmos de aprendizaje automático aplicados a problemas en ciberseguridad.

"""

import numpy as np
import os

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, MaxPool2D, Flatten
from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import CSVLogger, EarlyStopping
from PIL import Image
from data import generar_data
from confusionKDD import confusion_matrix_pablo


def crear_cnn(input_shape: tuple, num_classes: int) -> Sequential:
    """
    Crea un modelo de red neuronal convolucional (CNN) para la detección de intrusiones.

    La arquitectura del modelo incluye dos capas convolucionales seguidas de max-pooling,
    y capas densas para la clasificación final.

    Args:
        input_shape (tuple): La forma de la entrada de la imagen en el formato (altura, ancho, canales).
        num_classes (int): El número de clases en el problema de clasificación.

    Returns:
        Sequential: El modelo CNN configurado para la detección de intrusiones.
    """
    model = Sequential()

    # Primera capa convolucional 2D
    model.add(Conv2D(filters=64, kernel_size=(3, 3), activation='relu', input_shape=input_shape))
    model.add(MaxPool2D(pool_size=(2, 2)))

    # Segunda capa convolucional 2D
    model.add(Conv2D(filters=128, kernel_size=(3, 3), activation='relu'))
    model.add(MaxPool2D(pool_size=(2, 2)))

    # Aplanar y añadir capas completamente conectadas
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dense(64, activation='relu'))
    model.add(Dense(num_classes, activation='softmax'))

    model.summary()
    return model


def visualize_image(image: np.ndarray, label: int, index: int) -> None:
    """
    Guarda una imagen en un archivo en la carpeta 'img' para visualización.

    La función convierte la matriz bidimensional en una imagen para ver como se visualizaría.

    Args:
        image (np.ndarray): Imagen a guardar, en formato (altura, ancho, canales).
        label (int): Etiqueta de la imagen.
        index (int): Índice de la imagen para crear un nombre único.

    Returns:
        None: La función guarda la imagen en un archivo en la carpeta 'img'.
    """
    img_folder = "img"
    if not os.path.exists(img_folder):
        os.makedirs(img_folder)

    # Eliminar cualquier dimensión extra
    image = np.squeeze(image)

    # Si la imagen es en escala de grises (2D), convertirla a RGB (3D)
    if len(image.shape) == 2:
        image = np.stack((image,)*3, axis=-1)

    # Verificar y ajustar la forma de la imagen
    if image.shape[-1] == 1:  # Si la última dimensión es 1, replicar para obtener 3 canales
        image = np.concatenate([image]*3, axis=-1)

    # Convertir la imagen a un objeto PIL
    image_pil = Image.fromarray((image * 255).astype(np.uint8))

    # Define el nombre del archivo de la imagen
    nombre = os.path.join(img_folder, f'{label}_{index}.png')

    # Guarda la imagen en la carpeta 'img'
    image_pil.save(nombre)
    
if __name__ == "__main__":
    # Hiperparámetros
    file_name = "KDD_CNN"
    input_shape = (11, 11, 1)  # Tamaño de la secuencia de entrada (ajustar según tus datos)
    num_classes = 5  # Número de clases para clasificación multiclase
    epochs = 30
    batch_size = 32

    # Preprocesamiento de los datos
    data,labels,images = generar_data()
    X_train, X_rest, y_train, y_rest = train_test_split(images, labels, test_size=0.25, random_state=0)
    X_val, X_test, y_val, y_test = train_test_split(X_rest, y_rest, test_size=0.6, random_state=1)
    y_train_numeric = np.argmax(y_test, axis=1)

    # Compilación y entrenamiento de la CNN
    csv_logger = CSVLogger('KDD_CNN.csv', append=False)
    early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
    
    cnn = crear_cnn(input_shape, num_classes)
    cnn.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    cnn.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, validation_data=(X_val, y_val), 
             callbacks=[csv_logger, early_stopping], shuffle=True)

    cnn.save(file_name + '.keras')

    # Evaluación del modelo
    confusion_matrix_pablo('KDD_CNN.keras', X_test, y_test, 'confusionMatrixCNN')
                
                
    
    






