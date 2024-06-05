
"""
Nombre del codigo: Modelo de CNN de reconocimiento de malware.
Base de datos: Microsoft Malware Dataset
Alumno: Jim√©nez Poyatos, Pablo

Script solo con el modelo. Nada de representaci√≥n de datos ni nada. Adem√°s el codigo apilado en funciones.

Para crear el modelo, he necesitado instalarme diferentes bibliotecas como numpy, tensorflow, keras, etc.

Adem√°s, he tenido que descargarme las imagenes de entrenamiento en diferentes carpetas (cada clase en una carpeta) 
y guardarlas en la misma carpeta donde estaba este script.
"""


# Importamos las bibliotecas necesarias
import numpy as np                                     # Importamos la biblioteca NumPy para operaciones num√©ricas eficientes
import pandas as pd                                    # Importamos la biblioteca pandas para manipulaci√≥n y an√°lisis de datos
import tensorflow as tf                                # Importamos TensorFlow, una biblioteca para aprendizaje autom√°tico
import os                                              # Importamos el m√≥dulo os para interactuar con el sistema operativo
from PIL import Image                                  # Importamos la clase Image del m√≥dulo PIL para trabajar con im√°genes
from sklearn.model_selection import train_test_split   # Importamos train_test_split para dividir los datos en conjuntos de entrenamiento y prueba
from keras.utils import to_categorical                 # Importamos to_categorical para codificar las etiquetas en formato one-hot
from keras.models import Sequential, load_model                   # Importamos Sequential, un modelo lineal para apilar capas de red neuronal
from keras.layers import Conv2D, MaxPool2D, Dense, Flatten, Dropout  # Importamos capas espec√≠ficas para construir una red neuronal convolucional
import matplotlib.pyplot as plt
import csv

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

def load_and_preprocess_data1(clas_min, clas_max):
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
    resultados = p.starmap(load_and_preprocess_data1, args_list)
    data = concat(list(map(first,resultados)))
    label = concat(list(map(second,resultados)))
    data = np.array(data)
    label = np.array(label)
    return data, label

      


# Definimos una funci√≥n para dividir y codificar los datos
def split_and_encode_data(data, labels, test_size=0.1, random_state=10):
    # Dividimos los datos en conjuntos de entrenamiento y prueba, y codificamos las etiquetas en formato one-hot
    X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=test_size, random_state=random_state) # Dividimos los conjuntos de datos data(array de dimensiones (39209, 30, 30, 3) (numero de muestras, dimension imagenes y numero de canales de colores RGB) y labels en conjunto de entrenamiento y de prueba. Las x hacen referencia a los datos y las y a las etiquetas. Le ponemos que se reparta en %. El porcentaje de datos para el test es el numero que le introduzcamos como parámatro de entrada.
    y_train = to_categorical(y_train, 43)  # Codificamos en one-hot las etiquetas de entrenamiento
    y_test = to_categorical(y_test, 43)    # Codificamos en one-hot las etiquetas de prueba
    return X_train, X_test, y_train, y_test
'''
# Definimos una funci√≥n para construir el modelo de red neuronal convolucional
def build_model(input_shape):
    model = Sequential()

    # Primera capa: 2 capas convolution + relu 224x224x64
    model.add(Conv2D(filters=64, kernel_size=(3, 3), activation='relu', input_shape=input_shape))
    model.add(Conv2D(filters=64, kernel_size=(3, 3), activation='relu'))

    # Segunda capa: 1 max pooling 112x112x128
    model.add(MaxPool2D(pool_size=(2, 2)))

    # Tercera capa: 2 capas convolution + relu 112x112x128
    model.add(Conv2D(filters=128, kernel_size=(3, 3), activation='relu'))
    model.add(Conv2D(filters=128, kernel_size=(3, 3), activation='relu'))

    # Cuarta capa: 1 max pooling 56x56x256
    model.add(MaxPool2D(pool_size=(2, 2)))

    # Quinta capa: 3 capas convolution + relu 56x56x256
    model.add(Conv2D(filters=256, kernel_size=(3, 3), activation='relu'))
    model.add(Conv2D(filters=256, kernel_size=(3, 3), activation='relu'))
    model.add(Conv2D(filters=256, kernel_size=(3, 3), activation='relu'))

    # Sexta capa: 1 max pooling 28x28x512
    model.add(MaxPool2D(pool_size=(2, 2)))

    # Séptima capa: 3 capas convolution + relu 28x28x512
    model.add(Conv2D(filters=512, kernel_size=(3, 3), activation='relu'))
    model.add(Conv2D(filters=512, kernel_size=(3, 3), activation='relu'))
    model.add(Conv2D(filters=512, kernel_size=(3, 3), activation='relu'))

    # Octava capa: 1 max pooling 14x14x512
    model.add(MaxPool2D(pool_size=(2, 2)))
    
    # Novena capa: 3 capas convolution + relu 14x14x512
    model.add(Conv2D(filters=512, kernel_size=(3, 3), activation='relu'))
    model.add(Conv2D(filters=512, kernel_size=(3, 3), activation='relu'))
    model.add(Conv2D(filters=512, kernel_size=(3, 3), activation='relu'))

    # Décima capa: 1 max pooling 7x7x512
    model.add(MaxPool2D(pool_size=(2, 2)))

    # Undécima capa: 2 fully connected + relu 1x1x4096
    model.add(Flatten())
    model.add(Dense(4096, activation='relu'))
    model.add(Dropout(rate=0.5))
    model.add(Dense(4096, activation='relu'))

    # Duodécima capa: 1 fully connected + relu 1x1x9
    model.add(Dense(9, activation='relu'))

    # Trigésima capa: 1 softmax
    model.add(Dense(9, activation='softmax'))

    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    model.summary()
    return model
'''

def create_convnet(input_shape, num_classes):
    model = Sequential()

    # Primera capa convolucional
    model.add(Conv2D(64, (3, 3), activation='relu', input_shape=input_shape))
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(MaxPool2D((2, 2)))

    # Segunda capa convolucional
    model.add(Conv2D(128, (3, 3), activation='relu'))
    model.add(Conv2D(128, (3, 3), activation='relu'))
    model.add(MaxPool2D((2, 2)))

    # Tercera capa convolucional
    model.add(Conv2D(256, (3, 3), activation='relu'))
    model.add(Conv2D(256, (3, 3), activation='relu'))
    model.add(Conv2D(256, (1, 1), activation='relu'))
    model.add(MaxPool2D((2, 2)))

    # Cuarta capa convolucional
    model.add(Conv2D(512, (3, 3), activation='relu'))
    model.add(Conv2D(512, (3, 3), activation='relu'))
    model.add(Conv2D(512, (1, 1), activation='relu'))
    model.add(MaxPool2D((2, 2)))

    # Quinta capa convolucional
    model.add(Conv2D(512, (3, 3), activation='relu'))
    model.add(Conv2D(512, (3, 3), activation='relu'))
    model.add(Conv2D(512, (1, 1), activation='relu'))
    model.add(MaxPool2D((2, 2)))

    # Capas totalmente conectadas
    model.add(Flatten())
    model.add(Dense(4096, activation='relu'))
    model.add(Dense(4096, activation='relu'))
    model.add(Dense(num_classes, activation='softmax'))
    
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    model.summary()

    return model


# Definimos una funci√≥n para entrenar el modelo
def train_model(model, file_name, X_train, y_train, X_test, y_test, batch_size=32, epochs=1):
    with tf.device('/GPU:0'):   # Indicamos que el entrenamiento del modelo se realizar√° en la GPU si est√° disponible; si no, en la CPU
        model.fit(X_train, y_train, batch_size=batch_size, epochs=epochs, validation_data=(X_test, y_test))
    model.save(file_name)       # Guardamos el modelo entrenado en un archivo

'''
La diferencia principal entre realizar el entrenamiento del modelo en la GPU o en la CPU radica en el rendimiento y la velocidad de entrenamiento.

GPU (Unidad de Procesamiento Gr√°fico):

Ventajas:
Las GPU est√°n dise√±adas espec√≠ficamente para manejar operaciones matriciales y paralelas, que son comunes en el entrenamiento de modelos de redes neuronales.
Pueden realizar c√°lculos en paralelo en grandes cantidades de datos, lo que acelera significativamente el entrenamiento de modelos, especialmente en tareas intensivas en c√°lculos, como las redes neuronales profundas.
Ofrecen un rendimiento superior en comparaci√≥n con las CPU para tareas de aprendizaje profundo.

Desventajas:
Pueden ser m√°s costosas y consumir m√°s energ√≠a que las CPU.
Puede haber limitaciones en la cantidad de memoria de la GPU disponible, lo que podr√≠a ser un factor en modelos muy grandes.

CPU (Unidad Central de Procesamiento):

Ventajas:
Disponibles en la mayor√≠a de las computadoras y servidores sin necesidad de hardware adicional.
Adecuadas para tareas generales de prop√≥sito m√∫ltiple y no solo para aprendizaje profundo.
Pueden ser m√°s econ√≥micas en t√©rminos de hardware y consumo de energ√≠a.
Desventajas:
Las CPU no est√°n dise√±adas espec√≠ficamente para tareas de aprendizaje profundo y pueden ser menos eficientes en t√©rminos de velocidad para ciertos tipos de operaciones, especialmente en modelos grandes.

'''

if __name__ == "__main__":
  
    # Cargamos los datos
    data, labels = load_and_preprocess_data_paralelo(43, 5)
     # Calcular la cantidad de datos por etiqueta
    label_counts = np.bincount(labels)

    # Crear un gr√°fico de barras con etiquetas
    plt.bar(range(len(label_counts)), label_counts, tick_label=range(len(label_counts)))

    # Agregar etiquetas con los n√∫meros en cada barra
    for i, count in enumerate(label_counts):
        plt.text(i, count + 0.1, str(count), ha='center', va='bottom')

    plt.xlabel('Etiquetas')
    plt.ylabel('Cantidad de Datos')
    plt.title('Cantidad de Datos por Etiqueta')
    plt.show()

    # Dividimos los datos
    X_train, X_test, y_train, y_test = split_and_encode_data(data, labels)
    
    # Construimos el modelo de CNN
    model = create_convnet(X_train.shape[1:],43) #.shape te dice las dimensiones y caracteristicas de X_train. La primera indica el numero que hay, las siguientes las dimensiones y los canales de colores (que es lo que nos interesa)
    
    # Entrenamos el modelo y lo guardamos en un archivo
    train_model(model, 'classifierPrueba.keras', X_train, y_train, X_test, y_test, 32, 5)
    


"""
Para predecir la se√±al que aparece en una imagen, tenemos que cargar el modelo despues de 20 epochs y
la imagen. Esta √∫ltima tenemos que pasarla a una imagen con tres canales  y redimensionarla a 30 x 30 pixeles.
Despues de que mi modelo prediga cual ser√≠a su etiqueta, le asignamos su nombre correspondiente del diccionario 
clases.
"""

def predecir(model, nombre):
    os.getcwd()
    # Cargar el modelo previamente entrenado
    model = load_model(model)
    
    # Cargar la imagen
    image_path = nombre 
    image0 = Image.open(image_path)
    image1 = image0.resize((224, 224))
    image2 = image1.convert('RGB')
    image3 = np.array(image2)
    # Normalizar los valores de p√≠xeles
    
    # Realizar la predicci√≥n con el modelo cargado
    prediction = model.predict(np.array([image3]))  # Asegurarse de que sea un arreglo de forma (1, 30, 30, 3)
    
    # Obtener la etiqueta predicha (√≠ndice de la clase con mayor probabilidad)
    predicted_class = np.argmax(prediction)
    
    # Crear un diccionario que mapea las clases a sus etiquetas
    clases = { 
        0: 'L√≠mite de velocidad (20 km/h)',
        1: 'L√≠mite de velocidad (30 km/h)', 
        2: 'L√≠mite de velocidad (50 km/h)', 
        3: 'L√≠mite de velocidad (60 km/h)', 
        4: 'L√≠mite de velocidad (70 km/h)', 
        5: 'L√≠mite de velocidad (80 km/h)', 
        6: 'Fin del l√≠mite de velocidad (80 km/h)', 
        7: 'L√≠mite de velocidad (100 km/h)', 
        8: 'L√≠mite de velocidad (120 km/h)', 
        9: 'Prohibido adelantar', 
        10: 'Prohibido adelantar veh√≠culos de m√°s de 3.5 toneladas', 
        11: 'Derecho de paso en intersecci√≥n', 
        12: 'Carretera con prioridad', 
        13: 'Ceder el paso', 
        14: 'Detenerse', 
        15: 'Prohibido el paso de veh√≠culos', 
        16: 'Prohibido el paso de veh√≠culos de m√°s de 3.5 toneladas',
        17: 'Prohibido el acceso', 
        18: 'Precauci√≥n general', 
        19: 'Curva peligrosa a la izquierda', 
        20: 'Curva peligrosa a la derecha', 
        21: 'Curva doble', 
        22: 'Carretera con baches', 
        23: 'Carretera resbaladiza', 
        24: 'Carretera se estrecha a la derecha', 
        25: 'Trabajo en la carretera', 
        26: 'Se√±ales de tr√°fico', 
        27: 'Peatones', 
        28: 'Cruce de ni√±os', 
        29: 'Cruce de bicicletas', 
        30: 'Precauci√≥n: hielo/nieve',
        31: 'Cruce de animales salvajes', 
        32: 'Fin de l√≠mites de velocidad y adelantamiento', 
        33: 'Girar a la derecha', 
        34: 'Girar a la izquierda', 
        35: 'Solo adelante', 
        36: 'Ir recto o girar a la derecha', 
        37: 'Ir recto o girar a la izquierda', 
        38: 'Mantenerse a la derecha', 
        39: 'Mantenerse a la izquierda', 
        40: 'Circulaci√≥n obligatoria en rotonda', 
        41: 'Fin de la prohibici√≥n de adelantar', 
        42: 'Fin de la prohibici√≥n de adelantar veh√≠culos de m√°s de 3.5 toneladas'
    }
    
    # Obtener la etiqueta correspondiente a la clase predicha
    predicted_label = clases[predicted_class]
    
    probabilities = prediction[0]
    
    # Imprimir la etiqueta predicha
    print(f'Clase predicha: {predicted_label}')
    
    print('Probabilidades:')
    for i, prob in enumerate(probabilities):
        print(f'{clases[i]}: {prob:.4f}')