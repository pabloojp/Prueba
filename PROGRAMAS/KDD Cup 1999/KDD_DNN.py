"""
Nombre del código: Modelo de DNN para resolver el problema de detección de intrusiones.

Alumno: Jimenez Poyatos, Pablo
Trabajo: Algoritmos de aprendizaje automático aplicados a problemas en ciberseguridad.

"""

import numpy as np

from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import CSVLogger, EarlyStopping
from data import generar_data
from confusionKDD import confusion_matrix_pablo

def create_dnn(input_shape: int, num_classes: int) -> Sequential:
    """
    Crea un modelo de red neuronal profunda (DNN) para la detección de intrusiones.

    La arquitectura del modelo incluye una capa de entrada, dos capas ocultas con funciones de activación ReLU,
    y una capa de salida con función de activación Softmax para la clasificación multiclase.

    Args:
        input_shape (int): El número de características de entrada para el modelo.
        num_classes (int): El número de clases para la clasificación multiclase.

    Returns:
        Sequential: El modelo DNN configurado para la detección de intrusiones.
    """
    model = Sequential()

    # Capa de entrada (122 características)
    model.add(Dense(122, input_dim=input_shape, activation='relu'))

    # Primera capa oculta (50 neuronas)
    model.add(Dense(50, activation='relu'))

    # Segunda capa oculta (30 neuronas)
    model.add(Dense(30, activation='relu'))

    # Capa de salida con función Softmax para la clasificación multiclase
    model.add(Dense(num_classes, activation='softmax'))
    
    model.summary()
    return model


if __name__ == "__main__":
    # Hiperparámetros iniciales
    file_name = "KDD_DNN"
    input_dim = 122
    num_classes = 5
    batch_size = 32
    epochs = 30
    learning_rate = 0.001  # Esto se puede ajustar según sea necesario

    #Preprocesamiento de datos
    data,labels,_ = generar_data()
    data = data.reshape((data.shape[0], data.shape[1])).astype(np.float32)
    X_train, X_rest, y_train, y_rest = train_test_split(data, labels, test_size=0.25, random_state=0)
    X_val, X_test, y_val, y_test = train_test_split(X_rest, y_rest, test_size=0.6, random_state=1)
    
    # Construir el modelo
    csv_logger = CSVLogger('KDD_DNN.csv', append=False)
    early_stopping = EarlyStopping(monitor='val_loss', patience=5,
                                   restore_best_weights=True)
    dnn = create_dnn(input_dim, num_classes)

    # Entrenar y compilar el modelo
    dnn.compile(optimizer=Adam(learning_rate=learning_rate), loss="categorical_crossentropy", metrics=['accuracy'])
    dnn.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, validation_data=(X_val, y_val), callbacks=[csv_logger, early_stopping], shuffle=True)
    dnn.save(file_name + '.keras')

    # Evaluar el modelo
    confusion_matrix_pablo('KDD_DNN.keras', X_test, y_test, 'confusionMatrixDNN')

    






