"""
Nombre del codigo: Modelo de RNN para resolver el problema de detección de intrusiones.

Alumno: Jimenez Poyatos, Pablo
Trabajo: Algoritmos de aprendizaje automático aplicados a problemas en ciberseguridad.

"""

import numpy as np

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.optimizers import SGD
from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import CSVLogger, EarlyStopping
from data import generar_data
from confusionKDD import confusion_matrix_pablo

def create_LSTM(hidden_nodes, output_dim, X_train):
    """
    Crea un modelo de red neuronal recurrente (RNN) utilizando LSTM para la detección de intrusiones.

    La arquitectura del modelo incluye una capa LSTM seguida de una capa densa para la clasificación multiclase.

    Args:
        hidden_nodes (int): Número de nodos en la capa LSTM.
        output_dim (int): Número de clases para la clasificación multiclase.
        X_train (np.ndarray): Datos de entrenamiento para obtener la forma de entrada (timesteps, features).

    Returns:
        Sequential: El modelo LSTM configurado para la detección de intrusiones.
    """

    model = Sequential()
    model.add(LSTM(hidden_nodes, input_shape=(X_train.shape[1], X_train.shape[2])))
    model.add(Dense(output_dim, activation='softmax'))
    model.summary()
    return model


if __name__ == "__main__":
    # Hiperparámetros iniciles
    file_name = "KDD_RNN"
    input_dim = 122
    output_dim = 5
    timesteps = 100
    batch_size = 50
    epochs = 50
    learning_rate = 0.001
    hidden_nodes = 80

    #Preprocesamiento de los datos
    data,labels,_ = generar_data()
    data.astype(np.float32)
    data = np.reshape(data, (data.shape[0], 1, data.shape[1]))
    X_train, X_rest, y_train, y_rest = train_test_split(data, labels, test_size=0.25, random_state=0)
    X_val, X_test, y_val, y_test = train_test_split(X_rest, y_rest, test_size=0.6, random_state=1)


    # Construir y entrenar el LSTM
    csv_logger = CSVLogger('KDD_RNN.csv', append=False)
    early_stopping = EarlyStopping(monitor='val_loss', patience=5,
                                   restore_best_weights=True)
    lstm = create_LSTM(hidden_nodes, output_dim, X_train)
    optimizer = SGD(learning_rate=learning_rate)
    lstm.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])
    lstm.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, validation_data=(X_val, y_val), 
             callbacks=[csv_logger, early_stopping], shuffle=True)
    lstm.save(file_name + '.keras')

    # Evaluación del modelo
    confusion_matrix_pablo('KDD_RNN.keras', X_test, y_test, 'confusionMatrixRNN')

