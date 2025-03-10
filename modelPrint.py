import tensorflow as tf

# Intenta cargar el modelo
model = tf.keras.models.load_model('cifar10_model.h5')
print(model.summary())