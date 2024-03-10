
"""
Nombre del codigo: Modelo de CNN de reconocimiento de se√±ales.
Guiado por: Tutorial de Kaggle (acceso al enlace el 12 de noviembre)
        https://www.kaggle.com/code/yacharki/traffic-signs-image-classification-96-cnn#6.-Training-the-Model
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



# Definimos una funci√≥n para cargar y procesar datos
def load_and_preprocess_data(path):
    """

    Returns
    -------
    data : TYPE
        DESCRIPTION.
    labels : TYPE
        DESCRIPTION.

    """
    os.getcwd() 
    data = []                                         # Lista para almacenar datos de im√°genes
    labels = []                                       # Lista para almacenar etiquetas de im√°genes
    classes = 43                                      # N√∫mero total de clases en el conjunto de datos (las 43 se√±ales diferentes)

    # Recorremos cada clase
    for i in range(classes):
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
                labels.append(i)                     # Agregamos la etiqueta correspondiente a la lista de etiquetas
            except:
                pass

    data = np.array(data)      # Convertimos la lista de datos a un array de NumPy para que sea más eficiente a la hora de manipular sus datos
    labels = np.array(labels)  # Convertimos la lista de etiquetas a un array de NumPy
    return data, labels

# Definimos una funci√≥n para dividir y codificar los datos
def split_and_encode_data(data, labels, test_size=0.2, random_state=42):
    # Dividimos los datos en conjuntos de entrenamiento y prueba, y codificamos las etiquetas en formato one-hot
    X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=test_size, random_state=random_state) # Dividimos los conjuntos de datos data(array de dimensiones (39209, 30, 30, 3) (numero de muestras, dimension imagenes y numero de canales de colores RGB) y labels en conjunto de entrenamiento y de prueba. Las x hacen referencia a los datos y las y a las etiquetas. Le ponemos que se reparta en %. El porcentaje de datos para el test es el numero que le introduzcamos como parámatro de entrada.
    y_train = to_categorical(y_train, 43)  # Codificamos en one-hot las etiquetas de entrenamiento
    y_test = to_categorical(y_test, 43)    # Codificamos en one-hot las etiquetas de prueba
    return X_train, X_test, y_train, y_test

# Definimos una funci√≥n para construir el modelo de red neuronal convolucional
def build_model(input_shape):
    model = Sequential()                                                                           # Creamos un modelo secuencial

    model.add(Conv2D(filters=32, kernel_size=(5, 5), activation='relu', input_shape=input_shape))  # input_shape indica la forma de los datos de entrada.
    model.add(Conv2D(filters=32, kernel_size=(5, 5), activation='relu'))
    model.add(MaxPool2D(pool_size=(2, 2)))
    model.add(Dropout(rate=0.25))
    # Filtros de 32 canales: Los filtros son peque√±as matrices que se utilizan para escanear partes de la imagen de entrada. En este caso, se est√°n utilizando 32 filtros en ambas capas. Cada filtro es como una "lente" que busca patrones espec√≠ficos en la imagen.
    # Kernel de convoluci√≥n 5x5: El kernel de convoluci√≥n es una matriz que se desliza sobre la imagen para realizar una operaci√≥n matem√°tica llamada convoluci√≥n. En este caso, se est√° utilizando un kernel de 5x5, lo que significa que cada filtro analiza una regi√≥n de 5x5 p√≠xeles de la imagen en cada paso.
    # Activaci√≥n 'relu': 'relu' es una funci√≥n de activaci√≥n llamada Rectified Linear Unit. Despu√©s de que cada filtro haya realizado la convoluci√≥n con la regi√≥n de la imagen, se aplica la funci√≥n 'relu'. Esta funci√≥n es muy simple: si el valor de salida de la convoluci√≥n es positivo, se mantiene tal cual; si es negativo, se establece en cero. Esto introduce no linealidad en la red y permite que la CNN aprenda caracter√≠sticas m√°s complejas y patrones en los datos.
    # Entonces, en estas dos capas, cada uno de los 32 filtros busca patrones en regiones de 5x5 p√≠xeles de la imagen de entrada y aplica la funci√≥n 'relu' para resaltar las caracter√≠sticas relevantes. Estas operaciones se realizan para cada filtro, lo que permite que la CNN extraiga m√∫ltiples caracter√≠sticas de la imagen.
    # Capa MaxPool2D: La capa MaxPool2D es una capa de submuestreo que reduce la dimensionalidad de las caracter√≠sticas extra√≠das por las capas de convoluci√≥n. En esta configuraci√≥n, se utiliza una ventana de 2x2 p√≠xeles (pool_size=(2,2)). Lo que hace esta capa es examinar grupos de 2x2 p√≠xeles en las caracter√≠sticas de la capa anterior y retiene solo el valor m√°ximo de esos 4 p√≠xeles. Este proceso reduce la cantidad de informaci√≥n y c√°lculos en la red, lo que ayuda a evitar el sobreajuste y mejora el rendimiento computacional.
    # Capa Dropout: La capa Dropout es una t√©cnica de regularizaci√≥n. La regularizaci√≥n es una forma de prevenir el sobreajuste en la red. En esta capa, se establece una fracci√≥n de las unidades de la capa anterior en cero de manera aleatoria durante el entrenamiento. En este caso, el valor es 0.25, lo que significa que aproximadamente el 25% de las unidades de esta capa se establecer√°n en cero en cada paso de entrenamiento. Esto fuerza a la red a aprender de manera m√°s robusta y generalizable, ya que no puede depender demasiado de ninguna unidad espec√≠fica.

    model.add(Conv2D(filters=64, kernel_size=(3, 3), activation='relu'))
    model.add(Conv2D(filters=64, kernel_size=(3, 3), activation='relu'))
    model.add(MaxPool2D(pool_size=(2, 2)))
    model.add(Dropout(rate=0.25))
 
    model.add(Flatten())                               #  Esta capa toma las caracter√≠sticas generadas por las capas de convoluci√≥n y las convierte en un vector unidimensional.
    model.add(Dense(256, activation='relu'))           # Se agrega una capa con 256 neuronas y activacion relu. Esta capa es una capa completamente conectada y se utiliza para aprender patrones m√°s globales de las caracter√≠sticas extra√≠das por las capas de convoluci√≥n.
    model.add(Dropout(rate=0.5))
    model.add(Dense(43, activation='softmax'))         # Esta es la capa de salida que produce la distribuci√≥n de probabilidad de las 10 clases posibles.

    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    model.summary()  # Imprimimos un resumen de la arquitectura del modelo
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
'''
if __name__ == "__main__":
  
    # Cargamos los datos
    data, labels = load_and_preprocess_data()
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
    model = build_model(X_train.shape[1:]) #.shape te dice las dimensiones y caracteristicas de X_train. La primera indica el numero que hay, las siguientes las dimensiones y los canales de colores (que es lo que nos interesa)
    
    # Entrenamos el modelo y lo guardamos en un archivo
    train_model(model, 'traffic_classifier.keras', X_train, y_train, X_test, y_test, 32, 5)

'''
"""
Para predecir la se√±al que aparece en una imagen, tenemos que cargar el modelo despues de 20 epochs y
la imagen. Esta √∫ltima tenemos que pasarla a una imagen con tres canales  y redimensionarla a 30 x 30 pixeles.
Despues de que mi modelo prediga cual ser√≠a su etiqueta, le asignamos su nombre correspondiente del diccionario 
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

