import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

def grafico_datos():
    df = pd.read_csv("trainLabels.csv", index_col="Id")
    malware_typ = ["Ramnit", "Lollipop", "Kelihos_ver3", "Vundo", "Simda", "Tracur", "Kelihos_ver1", "Obfuscator.ACY", "Gatak"]
    malware_clasif = dict(zip(range(1,10), malware_typ))
    distribucion = []
    for i in range(1,10):
        contador = df.apply(lambda x: x['Class'] == i, axis=1).sum()
        distribucion.append([malware_clasif[i],contador])
    init = pd.DataFrame(distribucion)
    init.index = range(1, 10)
    print(init)
    init.plot(kind = 'bar',
              rot = 0,
              title = 'Distribución del training dataset (Microsoft BIG 2015)',
              color='#1f77b4',
              legend = False)            # Cambiamos el tamaño de la figura
    plt.savefig("distribMMD1.png")
    plt.show()


    # Parte 2 random:

    bucle = [(y_train,"train"), (y_test, "test"), (y_val,"val")]
    for i in bucle:
        distribu = pd.DataFrame([np.where(r==1)[0][0] for r in i[0]])
        distrib = []
        for j in range(9):
            contador = distribu.apply(lambda x: x[0] == j, axis=1).sum()
            distrib.append(contador)
        init[i[1]] = distrib

    init.columns = ["Tipo malware", "Etiquetas", "Train", "Test", "Validation"]
    init.index = range(1, 10)
    init.plot(kind='bar',
          rot=0,
          y=['Train', 'Test', 'Validation'],
          title='Distribución del training dataset (Microsoft BIG 2015)',
          legend=True)            # Cambiamos el tamaño de la figura
    plt.savefig("repartoMMD1.png")
    plt.show()
grafico_datos()
# DataFrame con los datos
'''
data = {
    "Tipo malware": ["Ramnit", "Lollipop", "Kelihos_ver3", "Vundo", "Simda", "Tracur", "Kelihos_ver1", "Obfuscator.ACY", "Gatak"],
    "Total": [1533, 2478, 2942, 475, 42, 751, 398, 1228, 1013],
    "Train": [1179, 1898, 2226, 383, 32, 574, 309, 942, 764],
    "Test": [212, 344, 379, 52,       6, 103, 53, 172, 146],
    "Validation": [142, 236, 337, 40, 4, 74, 36, 114, 103]
}

df = pd.DataFrame(data)

# Configuración del gráfico
plt.figure(figsize=(10, 10))

# Colores para el anillo exterior
#colors_outer = ['#ff9999','#66b3ff','#99ff99','#ffcc99', '#c2c2f0', '#ffb3e6', '#ff6666', '#c2f0c2', '#ffb3b3']
colors_inner = ['#001f3f', '#0074D9', '#6495ED',
                '#FF4500', '#FFA500', '#FFE4B5',
                '#006400', '#008000', '#00FF7F',
                '#8B0000', '#FF0000', '#FFA07A',
                '#8A2BE2', '#FF00FF', '#DA70D6',
                '#8B4513', '#CD853F', '#FFDAB9',
                '#FF69B4', '#FFC0CB', '#F5F5F5',
                '#000000', '#696969', '#D3D3D3',
                '#2F4F4F', '#9ACD32', '#FFFFE0']
# Datos para el anillo interior
inner_sizes = [1179, 212, 142, 1898, 344, 236, 2226, 379, 337, 383, 52, 40, 32, 6, 4, 574, 103, 74, 309, 53, 36, 942, 172, 114, 764, 146, 103]
inner_labels = ["Ramnit Train", "Ramnit Test", "Ramnit Validation",
                "Lollipop Train", "Lollipop Test", "Lollipop Validation",
                "Kelihos_ver3 Train", "Kelihos_ver3 Test", "Kelihos_ver3 Validation",
                "Vundo Train", "Vundo Test", "Vundo Validation",
                "Simda Train", "Simda Test", "Simda Validation",
                "Tracur Train", "Tracur Test", "Tracur Validation",
                "Kelihos_ver1 Train", "Kelihos_ver1 Test", "Kelihos_ver1 Validation",
                "Obfuscator.ACY Train", "Obfuscator.ACY Test", "Obfuscator.ACY Validation",
                "Gatak Train", "Gatak Test", "Gatak Validation"]

# Anillo exterior
outer_labels = df['Tipo malware']
outer_sizes = df['Total']
plt.pie(outer_sizes, labels=outer_labels, startangle=250)

# Anillo interior
plt.pie(inner_sizes, radius=0.6, startangle=250, colors = colors_inner)

plt.tight_layout()
plt.show()


darkblue, blue, cornflowerblue
darkorange, orange, moccasin
darkgreen, green, springgreen
darkred, red, lightsalmon
darkmagenta, magenta, orchid
saddlebrown, peru, peachpuff
hotpink, pink, lavenderblush
black, gray, white
darkslategrey, yellowgreen, lightyellow
'''

a=np.array([0,0,0,1,0,1,0,1])
print(np.where(a==1))
print(np.where(a==1)[0])
print(np.where(a==1)[0][0])