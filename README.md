<img align="center" src="https://i.imgur.com/ZgHWFhw.png" alt="gabriellugo" />

# IMAGE RECOGNITION

<a href="https://github.com/GabrielLugooo/Image-Recogn" target="_blank" rel="noreferrer noopener"> <img align="center" src="https://img.shields.io/badge/English%20Image%20Recognition-000000" alt="English Image Recogn" /></a>
<a href="https://github.com/GabrielLugooo/Image-Recogn/blob/main/README%20Spanish.md" target="_blank" rel="noreferrer noopener"> <img align="center" src="https://img.shields.io/badge/Spanish%20Image%20Recognition-green" alt="Spanish Image Recogn" /></a>

### Objective

This project aims to develop an image recognition application using `artificial intelligence`, based on a convolutional neural network `CNN` trained with the `CIFAR-10` dataset. The application allows to classify images into ten different categories, including airplanes, cars, birds, cats, dogs, deer, frogs, horses, boats and trucks. Through the use of data augmentation and normalization techniques, the performance of the model is optimized to improve its accuracy.

In addition, the application has a graphical interface created with `Streamlit`, where users can upload images and obtain predictions in real time. This development seeks to improve the understanding of computer vision models, their implementation in practical applications and the integration of deep learning models into interactive interfaces accessible to users.

### Skills Learned

- Implementation of convolutional neural networks `CNN` for image classification.
- Use of `TensorFlow` and `Keras` for model training and optimization.
- Image preprocessing and data augmentation techniques with `ImageDataGenerator`.
- Integration of the `AI` model with an interactive user interface in `Streamlit`.
- Implementation of `EarlyStopping` to optimize model training.
- Use of `PIL` and `Matplotlib` for image and result handling and visualization.

### Tools Used

![Static Badge](https://img.shields.io/badge/Python-000000?logo=python&logoSize=auto)
![Static Badge](https://img.shields.io/badge/Tensorflow-000000?logo=tensorflow&logoSize=auto)
![Static Badge](https://img.shields.io/badge/Keras-000000?logo=keras&logoSize=auto)
![Static Badge](https://img.shields.io/badge/OpenCV-000000?logo=opencv&logoSize=auto)
![Static Badge](https://img.shields.io/badge/Numpy-000000?logo=numpy&logoSize=auto)
![Static Badge](https://img.shields.io/badge/Streamlit-000000?logo=streamlit&logoSize=auto)

- `Python` (Main programming language of the project).
- `TensorFlow`/`Keras` Deep learning library used to build and train the convolutional neural network (CNN).
- `OpenCV` Library for image processing and computer vision.
- `NumPy` Library for matrix handling and essential mathematical operations in image preprocessing.
- `Matplotlib` Used for data visualization and model prediction graphs.
- `Streamlit` Framework to create the interactive graphical interface of the application.
- `PIL` (Python Imaging Library) Library used for image manipulation in the application.

### Project

#### Preview

<img align="center" src="https://i.imgur.com/NXFugzi.jpeg" alt="ImageRecogn 01" />
<img align="center" src="https://i.imgur.com/HTEL5LP.jpeg" alt="ImageRecogn 02" />
<img align="center" src="https://i.imgur.com/IAl5AVN.jpeg" alt="ImageRecogn 03" />

#### Code with Comments (English)

- **Model CIFAR10**

```python
# Model Recognition

# Import the necessary libraries
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' # 0 = DEBUG, 1 = INFO, 2 = WARNING, 3 = ERROR

import sys
import site
print(sys.executable) # Check which Python interpreter is being used
print(site.getsitepackages()) # Show where to look for packages

import tensorflow as tf
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Flatten, Dense, Conv2D, MaxPooling2D
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping

# Load CIFAR-10 data
(x_train, y_train), (x_test, y_test) = cifar10.load_data()

# Normalize images
x_train = x_train / 255.0
x_test = x_test / 255.0

# Convert labels to categories
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

# Define the model cnn
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

# Compile the model
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# Define EarlyStopping
early_stopping = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)

# Adjust the model with data augmentation and EarlyStopping
model.fit(datagen.flow(x_train, y_train, batch_size=64),
          epochs=10,
          validation_data=(x_test, y_test),
          callbacks=[early_stopping])

# Save the trained model
model.save('cifar10_model.h5')
```

- **Model Test**

```python
import tensorflow as tf

# Try to load the model
model = tf.keras.models.load_model('cifar10_model.h5')
print(model.summary())
```

- **Image Recognition AI App**

```python
# Image Recognition AI APP

# Import the necessary libraries
import numpy as np
import matplotlib.pyplot as plt
import streamlit as st
import tensorflow as tf
from PIL import Image

def main():
    st.title('Image Classifier')
    st.write('Upload an image and get the AI ​​prediction')

    file = st.file_uploader('Upload an image')

    if file:
        image = Image.open(file)
        st.image(image, use_column_width=True)

        # Preprocess the image
        resized_image = image.resize((32, 32))
        img_array = np.array(resized_image) / 255.0 # Normalize correctly (percentage between 0 and 1)
        img_array = img_array.reshape((1, 32, 32, 3)) # Resize the image for the model

        # Load the model and make the prediction
        model = tf.keras.models.load_model('cifar10_model.h5')
        predictions = model.predict(img_array)

        # CIFAR-10 class labels
        cifar10_classes = ['airplane', 'automobile', 'bird', 'cat', 'dog', 'deer', 'frog', 'horse', 'ship', 'truck']

        # Plot the prediction
        fig, ax = plt.subplots()
        ax.barh(cifar10_classes, predictions[0], align='center')
        ax.set_xlabel('Probability')
        ax.set_title('Image Prediction')

        # Display the graph in Streamlit
        st.pyplot(fig)

    else:
        st.text('You have not loaded an image')

if __name__ == "__main__":
    main()
```

### Limitations

Image Recognithion it's just for educational purpose under the MIT License.
The code for the minimal version is also available.

---

<h3 align="left">Connect with me</h3>

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
<a href="https://github.com/GabrielLugooo/GabrielLugooo/blob/main/README.md" target="_blank" rel="noreferrer noopener"> <img align="center" src="https://img.shields.io/badge/English%20Version-000000" alt="English Version" /></a>
<a href="https://github.com/GabrielLugooo/GabrielLugooo/blob/main/Readme%20Spanish.md" target="_blank" rel="noreferrer noopener"> <img align="center" src="https://img.shields.io/badge/Spanish%20Version-Green" alt="Spanish Version" /></a>
</p>

<a href="https://linktr.ee/gabriellugooo" target="_blank" rel="noreferrer noopener"> <img align="center" src="https://img.shields.io/badge/Credits-Gabriel%20Lugo-green" alt="Credits" /></a>
<img align="center" src="https://komarev.com/ghpvc/?username=GabrielLugoo&label=Profile%20views&color=green&base=2000" alt="GabrielLugooo" />
<a href="" target="_blank" rel="noreferrer noopener"> <img align="center" src="https://img.shields.io/badge/License-MIT-green" alt="MIT License" /></a>
