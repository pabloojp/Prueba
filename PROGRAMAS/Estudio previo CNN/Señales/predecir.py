"""
Nombre del codigo: Prediccion señal de trafico usando el modelo creado. 
Alumno: Jiménez Poyatos, Pablo

Para predecir la señal que aparece en una imagen, tenemos que cargar el modelo después de 20 epochs y
la imagen. Esta última tenemos que pasarla a una imagen con tres canales y redimensionarla a 30 x 30 píxeles.
Después de que mi modelo prediga cual sería su etiqueta, le asignamos su nombre correspondiente del diccionario
clases.
"""

from PIL import Image  # Importamos la biblioteca Pillow para manipulación de imágenes
import numpy as np  # Importamos NumPy para operaciones con arrays
from keras.models import load_model  # Importamos load_model de Keras para cargar modelos preentrenados

# Definimos la función para predecir la señal de tráfico
def predict_traffic_sign(image_path, model_path):
    """
    Predice la señal de tráfico en la imagen proporcionada utilizando el modelo de CNN preentrenado.

    Parámetros:
        image_path (str): La ruta de la imagen que se va a predecir.
        model_path (str): La ruta del archivo del modelo preentrenado.

    Retorna:
        str: La etiqueta predicha para la imagen.
        dict: Un diccionario con las probabilidades para cada clase.
    """
    # Cargar el modelo previamente entrenado
    model = load_model(model_path)

    # Cargar la imagen
    image = Image.open(image_path)
    image = image.resize((30, 30))  # Redimensionar la imagen a 30x30 píxeles
    image = image.convert('RGB')  # Convertir la imagen a formato RGB
    image = np.array(image)  # Convertir la imagen a un array NumPy
    image = image / 255.0  # Normalizar los valores de píxeles entre 0 y 1

    # Realizar la predicción con el modelo cargado
    prediction = model.predict(np.array([image]))  # Asegurarse de que sea un arreglo de forma (1, 30, 30, 3)

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

    # Extraer probabilidades
    probabilities = {clases[i]: prob for i, prob in enumerate(prediction[0])}

    return predicted_label, probabilities

if __name__ == "__main__":
    # Ruta de la imagen a predecir
    image_path = 'prohib.jpg'

    # Ruta del modelo preentrenado
    model_path = 'traffic_classifier.keras'

    # Predecir la señal de tráfico
    predicted_label, probabilities = predict_traffic_sign(image_path, model_path)

    # Imprimir la etiqueta predicha
    print(f'Clase predicha: {predicted_label}')

    # Imprimir las probabilidades para cada clase
    print('Probabilidades:')
    for clase, prob in probabilities.items():
        print(f'{clase}: {prob:.4f}')
