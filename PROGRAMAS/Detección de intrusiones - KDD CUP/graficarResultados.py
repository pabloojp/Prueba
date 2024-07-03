'''
Función para graficar resultados recogidos en csv sobre la evolución del modelo a lo largo de las épocas.
'''

import pandas as pd
import matplotlib.pyplot as plt
import os
import numpy as np


def plot_training_history(csv_file, name_file):
    # Leer los datos del CSV
    data = pd.read_csv(csv_file)

    # Extraer los datos relevantes
    epochs = data['epoch']
    train_loss = data['loss']
    val_loss = data['val_loss']
    train_acc = data['accuracy']
    val_acc = data['val_accuracy']

    # Crear las gráficas
    plt.figure(figsize=(12, 4))
    plt.suptitle(name_file, fontsize=16)

    # Configuración de los ejes
    x_ticks = range(0, 26, 5)  # Ajustar según el número de épocas
    loss_y_ticks = [i / 10.0 for i in range(0, 25, 1)]  # Ajustar según los valores de loss
    acc_y_ticks = [i / 10.0 for i in range(4, 11, 1)]  # Ajustar según los valores de accuracy

    # Subplot 1: Training and validation loss
    plt.subplot(1, 2, 1)
    plt.plot(epochs, train_loss, label='Train Loss', color='blue')
    plt.plot(epochs, val_loss, label='Validation Loss', color='red')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()


    # Subplot 2: Training and validation accuracy
    plt.subplot(1, 2, 2)
    plt.plot(epochs, train_acc, label='Train Accuracy', color='blue')
    plt.plot(epochs, val_acc, label='Validation Accuracy', color='red')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('Training and Validation Accuracy')
    plt.legend()


    plt.tight_layout()
    # Verificar y añadir extensión si es necesario
    valid_extensions = ['.eps', '.jpeg', '.jpg', '.pdf', '.pgf', '.png', '.ps', '.raw', '.rgba', '.svg', '.svgz',
                        '.tif', '.tiff', '.webp']
    if not any(name_file.lower().endswith(ext) for ext in valid_extensions):
        name_file += '.png'

    plt.savefig(name_file)
    plt.show()

def plot_training_history_ae(csv_file, name_file):
    # Leer los datos del CSV
    data = pd.read_csv(csv_file)

    # Extraer los datos relevantes
    epochs = data['epoch']
    train_loss = data['loss']
    val_loss = data['val_loss']

    # Crear las gráficas
    plt.figure(figsize=(8, 6))
    plt.suptitle(name_file, fontsize=16)
    # Configuración de los ejes
    x_ticks = range(0, 100, 5)  # Ajustar según el número de épocas
    loss_y_ticks = np.arange(0, 0.06, 0.005).tolist()  # Ajustar según los valores de loss

    # Gráfico: Training and validation loss
    plt.plot(epochs, train_loss, label='Train Loss', color='blue')
    plt.plot(epochs, val_loss, label='Validation Loss', color='red')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    #plt.xticks(x_ticks)
    #plt.yticks(loss_y_ticks)  # Ajustar el límite de los valores de loss

    plt.tight_layout()
    # Verificar y añadir extensión si es necesario
    valid_extensions = ['.eps', '.jpeg', '.jpg', '.pdf', '.pgf', '.png', '.ps', '.raw', '.rgba', '.svg', '.svgz',
                        '.tif', '.tiff', '.webp']
    if not any(name_file.lower().endswith(ext) for ext in valid_extensions):
        name_file += '.png'

    plt.savefig(name_file)
    plt.show()

directorio = os.getcwd()

# Ir a una carpeta anterior
directorio_anterior = os.path.abspath(os.path.join(directorio, os.pardir))

# Entrar en una carpeta dentro de la carpeta anterior
carpeta_resultados = os.path.join(directorio_anterior)
archivos = os.listdir(carpeta_resultados)

for archivo in ["KDD_CNN.csv", "KDD_DNN.csv", "KDD_RNN.csv"]:
    ruta = os.path.join(carpeta_resultados, archivo)
    plot_training_history(ruta,archivo)
