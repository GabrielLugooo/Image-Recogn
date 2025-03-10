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



