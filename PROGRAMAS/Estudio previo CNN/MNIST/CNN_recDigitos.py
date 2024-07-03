"""
Nombre del código: Modelo CNN reconocimiento de dígitos usando dataset MNIST.
Guiado por: Tutorial de Kaggle (acceso al enlace el 10 de noviembre)
        https://www.kaggle.com/code/kanncaa1/convolutional-neural-network-cnn-tutorial
Alumno: Jiménez Poyatos, Pablo

Script solo con el modelo. Nada de representación de datos ni nada. Además el código apilado en funciones.

Para crear el modelo, he necesitado instalarme diferentes bibliotecas como numpy, tensorflow, keras, etc.

Además, he tenido que descargarme los datos de entrenamiento como archivos CSV y guardarlos en
la misma carpeta donde estaba este script.
"""

# Importación de bibliotecas:
import pandas as pd     # Para el análisis y manipulación de datos.
from sklearn.model_selection import train_test_split   # Para dividir el conjunto de datos en entrenamiento y evaluación.
import tensorflow as tf
from keras.utils import to_categorical                 # Para la codificación one-hot de etiquetas.
from keras.models import Sequential                    # Para construir el modelo secuencial.
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPool2D  # Para definir las capas de la CNN.
from keras.preprocessing.image import ImageDataGenerator # Para la generación de imágenes aumentadas.
from keras.callbacks import ReduceLROnPlateau          # Para reducir la tasa de aprendizaje si no mejora el rendimiento.

# Función para cargar los datos
def load_data(train_file: str, test_file: str) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Carga los datos de entrenamiento y prueba desde archivos CSV, normaliza y transforma las imágenes.

    Argumentos:
    - train_file: Ruta al archivo CSV de entrenamiento.
    - test_file: Ruta al archivo CSV de prueba.

    Devuelve:
    - X_train: Imágenes de entrenamiento normalizadas y transformadas.
    - Y_train: Etiquetas de entrenamiento codificadas en one-hot.
    - test: Imágenes de prueba normalizadas y transformadas.
    """
    train = pd.read_csv(train_file)
    test = pd.read_csv(test_file)

    Y_train = train["label"]
    X_train = train.drop(labels=["label"], axis=1)
    del train

    X_train = X_train / 255.0
    test = test / 255.0

    X_train = X_train.values.reshape(-1, 28, 28, 1)
    test = test.values.reshape(-1, 28, 28, 1)

    Y_train = to_categorical(Y_train, num_classes=10)

    return X_train, Y_train, test

# Función para dividir los datos en entrenamiento y validación
def split_data(X_train: pd.DataFrame, Y_train: pd.DataFrame, test_size: float = 0.1, random_seed: int = 2) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Divide los datos en conjuntos de entrenamiento y validación.

    Argumentos:
    - X_train: Imágenes de entrenamiento.
    - Y_train: Etiquetas de entrenamiento.
    - test_size: Proporción de datos para el conjunto de validación.
    - random_seed: Semilla para la aleatorización de la división.

    Devuelve:
    - X_train: Conjunto de entrenamiento.
    - X_val: Conjunto de validación.
    - Y_train: Etiquetas de entrenamiento.
    - Y_val: Etiquetas de validación.
    """
    X_train, X_val, Y_train, Y_val = train_test_split(X_train, Y_train, test_size=test_size, random_state=random_seed)
    return X_train, X_val, Y_train, Y_val

# Función para definir el modelo CNN
def create_model() -> Sequential:
    """
    Define y retorna un modelo CNN secuencial para la clasificación de dígitos.

    Devuelve:
    - model: El modelo CNN compilado.
    """
    model = Sequential()

    model.add(Conv2D(filters=32, kernel_size=(5, 5), padding='same', activation='relu', input_shape=(28, 28, 1)))
    model.add(Conv2D(filters=32, kernel_size=(5, 5), padding='same', activation='relu'))
    model.add(MaxPool2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Conv2D(filters=64, kernel_size=(3, 3), padding='same', activation='relu'))
    model.add(Conv2D(filters=64, kernel_size=(3, 3), padding='same', activation='relu'))
    model.add(MaxPool2D(pool_size=(2, 2), strides=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Flatten())
    model.add(Dense(256, activation="relu"))
    model.add(Dropout(0.5))
    model.add(Dense(10, activation="softmax"))

    model.summary()

    return model

# Función para compilar el modelo
def compile_model(model: Sequential) -> None:
    """
    Compila el modelo CNN con un optimizador y función de pérdida específicos.

    Argumentos:
    - model: El modelo CNN a compilar.

    Devuelve:
    - None
    """
    optimizer = tf.keras.optimizers.RMSprop(learning_rate=0.001, rho=0.9, epsilon=1e-08, centered=False)
    model.compile(optimizer=optimizer, loss="categorical_crossentropy", metrics=["accuracy"])

# Función para configurar el generador de imágenes aumentadas
def configure_image_generator(X_train: pd.DataFrame) -> ImageDataGenerator:
    """
    Configura y retorna un generador de imágenes aumentadas para el entrenamiento.

    Argumentos:
    - X_train: Imágenes de entrenamiento para el ajuste del generador.

    Devuelve:
    - datagen: Generador de imágenes aumentadas.
    """
    datagen = ImageDataGenerator(
        featurewise_center=False,
        samplewise_center=False,
        featurewise_std_normalization=False,
        samplewise_std_normalization=False,
        zca_whitening=False,
        rotation_range=10,
        zoom_range=0.1,
        width_shift_range=0.1,
        height_shift_range=0.1,
        horizontal_flip=False,
        vertical_flip=False)

    datagen.fit(X_train)
    return datagen

# Función para entrenar el modelo
def train_model(model: Sequential, datagen: ImageDataGenerator, X_train: pd.DataFrame, Y_train: pd.DataFrame, X_val: pd.DataFrame, Y_val: pd.DataFrame, epochs: int = 5, batch_size: int = 86) -> None:
    """
    Entrena el modelo CNN con los datos de entrenamiento y validación.

    Argumentos:
    - model: El modelo CNN a entrenar.
    - datagen: Generador de imágenes aumentadas.
    - X_train: Imágenes de entrenamiento.
    - Y_train: Etiquetas de entrenamiento.
    - X_val: Imágenes de validación.
    - Y_val: Etiquetas de validación.
    - epochs: Número de épocas de entrenamiento.
    - batch_size: Tamaño de lote para el entrenamiento.

    Devuelve:
    - None
    """
    learning_rate_reduction = ReduceLROnPlateau(monitor='val_accuracy', patience=3, verbose=1, factor=0.5, min_lr=0.00001)
    model.fit(datagen.flow(X_train, Y_train, batch_size=batch_size), epochs=epochs, validation_data=(X_val, Y_val), verbose=2, steps_per_epoch=X_train.shape[0] // batch_size, callbacks=[learning_rate_reduction])
    model.save('modelo_digitos_func.keras')

if __name__ == "__main__":
    # Cargar datos
    X_train, Y_train, test = load_data("train.csv", "test.csv")

    y_train_pandas = pd.Series(Y_train)

    y_train_pandas.value_counts()

    # Dividir datos
    X_train, X_val, Y_train, Y_val = split_data(X_train, Y_train)

    # Crear modelo
    model = create_model()

    # Compilar modelo
    compile_model(model)

    # Configurar generador de imágenes
    datagen = configure_image_generator(X_train)

    # Entrenar modelo
    train_model(model, datagen, X_train, Y_train, X_val, Y_val, epochs=20, batch_size=86)
