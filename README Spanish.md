<img align="center" src="https://media.licdn.com/dms/image/v2/D4D16AQGUNxQ7NSC05A/profile-displaybackgroundimage-shrink_350_1400/profile-displaybackgroundimage-shrink_350_1400/0/1738695150340?e=1749686400&v=beta&t=hBmszzzG0Zu-m7ZxeCdU5VxgDWqIZuWB0vnrMycuqY4" alt="gabriellugo" />

# RECONOCEDOR DE IMAGENES

<a href="https://github.com/GabrielLugooo/Image-Recogn/blob/main/README%20Spanish.md" target="_blank" rel="noreferrer noopener"> <img align="center" src="https://img.shields.io/badge/Reconocedor%20de%20Imagenes%20Español-000000" alt="Reconocedor de Imagenes Español" /></a>
<a href="https://github.com/GabrielLugooo/Image-Recogn" target="_blank" rel="noreferrer noopener"> <img align="center" src="https://img.shields.io/badge/Reconocedor%20de%20Imagenes%20Inglés-green" alt="Reconocedor de Imagenes Inglés" /></a>

### Objetivos

Este proyecto tiene como objetivo desarrollar una aplicación de reconocimiento de imágenes utilizando `inteligencia artificial`, basada en una red neuronal convolucional `CNN` entrenada con el conjunto de datos `CIFAR-10`. La aplicación permite clasificar imágenes en diez categorías diferentes, incluyendo aviones, automóviles, pájaros, gatos, perros, ciervos, ranas, caballos, barcos y camiones. A través del uso de técnicas de aumento de datos y normalización, se optimiza el rendimiento del modelo para mejorar su precisión.

Además, la aplicación cuenta con una interfaz gráfica creada con `Streamlit`, donde los usuarios pueden cargar imágenes y obtener predicciones en tiempo real. Este desarrollo busca mejorar la comprensión de modelos de visión por computadora, su implementación en aplicaciones prácticas y la integración de modelos de aprendizaje profundo en interfaces interactivas accesibles para los usuarios.

### Habilidades Aprendidas

- Implementación de redes neuronales convolucionales `CNN` para clasificación de imágenes.
- Uso de `TensorFlow` y `Keras` para el entrenamiento y optimización del modelo.
- Preprocesamiento de imágenes y técnicas de aumento de datos con `ImageDataGenerator`.
- Integración del modelo de `IA` con una interfaz de usuario interactiva en `Streamlit`.
- Implementación de `EarlyStopping` para optimizar el entrenamiento del modelo.
- Uso de `PIL` y `Matplotlib` para el manejo y visualización de imágenes y resultados.

### Herramientas Usadas

![Static Badge](https://img.shields.io/badge/Python-000000?logo=python&logoSize=auto)
![Static Badge](https://img.shields.io/badge/Tensorflow-000000?logo=tensorflow&logoSize=auto)
![Static Badge](https://img.shields.io/badge/Keras-000000?logo=keras&logoSize=auto)
![Static Badge](https://img.shields.io/badge/OpenCV-000000?logo=opencv&logoSize=auto)
![Static Badge](https://img.shields.io/badge/Numpy-000000?logo=numpy&logoSize=auto)
![Static Badge](https://img.shields.io/badge/Streamlit-000000?logo=streamlit&logoSize=auto)

- `Python` (Lenguaje de programación principal del proyecto).
- `TensorFlow`/`Keras` Biblioteca de aprendizaje profundo utilizada para construir y entrenar la red neuronal convolucional (CNN).
- `OpenCV` Biblioteca para el procesamiento de imágenes y visión por computadora.
- `NumPy` Biblioteca para manejo de matrices y operaciones matemáticas esenciales en el preprocesamiento de imágenes.
- `Matplotlib` Utilizada para la visualización de datos y gráficos de predicción del modelo.
- `Streamlit` Framework para crear la interfaz gráfica interactiva de la aplicación.
- `PIL` (Python Imaging Library) Biblioteca utilizada para la manipulación de imágenes en la aplicación.

### Proyecto

#### Vista Previa

<img align="center" src="https://i.imgur.com/NXFugzi.jpeg" alt="ImageRecogn 01" />
<img align="center" src="https://i.imgur.com/HTEL5LP.jpeg" alt="ImageRecogn 02" />
<img align="center" src="https://i.imgur.com/IAl5AVN.jpeg" alt="ImageRecogn 03" />

#### Código con Comentarios (Español)

- **Model CIFAR10**

```python
# Model Recognition

# Importar las librerías necesarias
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # 0 = DEBUG, 1 = INFO, 2 = WARNING, 3 = ERROR

import sys
import site
print(sys.executable)  # Verifica qué intérprete de Python está usando
print(site.getsitepackages())  # Muestra dónde busca los paquetes

import tensorflow as tf
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Flatten, Dense, Conv2D, MaxPooling2D
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping

# Cargar los datos de CIFAR-10
(x_train, y_train), (x_test, y_test) = cifar10.load_data()

# Normalizar las imágenes
x_train = x_train / 255.0
x_test = x_test / 255.0

# Convertir las etiquetas en categorías
y_train = to_categorical(y_train, 10)
y_test = to_categorical(y_test, 10)

# Data augmentation
datagen = ImageDataGenerator(
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)

# Definir el modelo CNN
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3)),
    MaxPooling2D(pool_size=(2, 2)),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D(pool_size=(2, 2)),
    Conv2D(64, (3, 3), activation='relu'),
    Flatten(),
    Dense(64, activation='relu'),
    Dense(10, activation='softmax')
])

# Compilar el modelo
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# Definir el EarlyStopping
early_stopping = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)

# Ajustar el modelo con data augmentation y EarlyStopping
model.fit(datagen.flow(x_train, y_train, batch_size=64),
          epochs=10,
          validation_data=(x_test, y_test),
          callbacks=[early_stopping])

# Guardar el modelo entrenado
model.save('cifar10_model.h5')
```

- **Model Test**

```python
import tensorflow as tf

# Intenta cargar el modelo
model = tf.keras.models.load_model('cifar10_model.h5')
print(model.summary())
```

- **Image Recognition AI App**

```python
# Image Recognition IA APP

# Importar las librerías necesarias
import numpy as np
import matplotlib.pyplot as plt
import streamlit as st
import tensorflow as tf
from PIL import Image

def main():
    st.title('Clasificador de Imagenes')
    st.write('Carga una imagen y obtendrás la predicción de la IA')

    file = st.file_uploader('Carga una imagen')

    if file:
        image = Image.open(file)
        st.image(image, use_column_width=True)

        # Preprocesar la imagen
        resized_image = image.resize((32, 32))
        img_array = np.array(resized_image) / 255.0  # Normalizar correctamente (porcentaje entre 0 y 1)
        img_array = img_array.reshape((1, 32, 32, 3))  # Redimensionar la imagen para el modelo

        # Cargar el modelo y hacer la predicción
        model = tf.keras.models.load_model('cifar10_model.h5')
        predictions = model.predict(img_array)

        # Etiquetas de las clases de CIFAR-10
        cifar10_classes = ['airplane', 'automobile', 'bird', 'cat', 'dog', 'deer', 'frog', 'horse', 'ship', 'truck']

        # Graficar la predicción
        fig, ax = plt.subplots()
        ax.barh(cifar10_classes, predictions[0], align='center')
        ax.set_xlabel('Probabilidad')
        ax.set_title('Predicción de la Imagen')

        # Mostrar la gráfica en Streamlit
        st.pyplot(fig)

    else:
        st.text('No has cargado una imagen')

if __name__ == "__main__":
    main()
```

### Limitaciones

El Reconocedor de Imagenes es solo para fines educativos bajo la licencia MIT.
También esta disponible el código de la versión mínimal.

---

<h3 align="left">Conecta Conmigo</h3>

<p align="left">
<a href="https://www.youtube.com/@gabriellugooo" target="_blank" rel="noreferrer noopener"> <img align="center" src="https://img.icons8.com/?size=50&id=55200&format=png" alt="@gabriellugooo" height="40" width="40" /></a>
<a href="http://www.tiktok.com/@gabriellugooo" target="_blank" rel="noreferrer noopener"> <img align="center" src="https://img.icons8.com/?size=50&id=118638&format=png" alt="@gabriellugooo" height="40" width="40" /></a>
<a href="https://instagram.com/lugooogabriel" target="_blank" rel="noreferrer noopener"> <img align="center" src="https://img.icons8.com/?size=50&id=32309&format=png" alt="lugooogabriel" height="40" width="40" /></a>
<a href="https://twitter.com/gabriellugo__" target="_blank" rel="noreferrer noopener"> <img align="center" src="https://img.icons8.com/?size=50&id=phOKFKYpe00C&format=png" alt="gabriellugo__" height="40" width="40" /></a>
<a href="https://www.linkedin.com/in/hernando-gabriel-lugo" target="_blank" rel="noreferrer noopener"> <img align="center" src="https://img.icons8.com/?size=50&id=8808&format=png" alt="hernando-gabriel-lugo" height="40" width="40" /></a>
<a href="https://github.com/GabrielLugooo" target="_blank" rel="noreferrer noopener"> <img align="center" src="https://img.icons8.com/?size=80&id=AngkmzgE6d3E&format=png" alt="gabriellugooo" height="34" width="34" /></a>
<a href="mailto:lugohernandogabriel@gmail.com"> <img align="center" src="https://img.icons8.com/?size=50&id=38036&format=png" alt="lugohernandogabriel@gmail.com" height="40" width="40" /></a>
<a href="https://linktr.ee/gabriellugooo" target="_blank" rel="noreferrer noopener"> <img align="center" src="https://simpleicons.org/icons/linktree.svg" alt="gabriellugooo" height="40" width="40" /></a>
</p>

<p align="left">
<a href="https://github.com/GabrielLugooo/GabrielLugooo/blob/main/Readme%20Spanish.md" target="_blank" rel="noreferrer noopener"> <img align="center" src="https://img.shields.io/badge/Versión%20Español-000000" alt="Versión Español" /></a>
<a href="https://github.com/GabrielLugooo/GabrielLugooo/blob/main/README.md" target="_blank" rel="noreferrer noopener"> <img align="center" src="https://img.shields.io/badge/Versión%20Inglés-Green" alt="Versión Inglés" /></a>

</p>

<a href="https://linktr.ee/gabriellugooo" target="_blank" rel="noreferrer noopener"> <img align="center" src="https://img.shields.io/badge/Créditos-Gabriel%20Lugo-green" alt="Créditos" /></a>
<img align="center" src="https://komarev.com/ghpvc/?username=GabrielLugoo&label=Vistas%20del%20Perfil&color=green&base=2000" alt="GabrielLugooo" />
<a href="" target="_blank" rel="noreferrer noopener"> <img align="center" src="https://img.shields.io/badge/License-MIT-green" alt="MIT License" /></a>
