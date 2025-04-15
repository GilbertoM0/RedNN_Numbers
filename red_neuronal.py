import tensorflow as tf
import tensorflow_datasets as tfds
import numpy as np
import math
datos, metadatos = tfds.load('mnist', as_supervised=True, with_info=True)
datos_entrenamiento = datos['train']
datos_prueba = datos['test']
def normalizar(imagen, etiqueta):
    imagen = tf.cast(imagen, tf.float32) / 255.0
    return imagen, etiqueta
datos_entrenamiento = datos_entrenamiento.map(normalizar).cache()
datos_prueba = datos_prueba.map(normalizar).cache()
modelo = tf.keras.Sequential([
    tf.keras.layers.Flatten(input_shape=(28, 28, 1)),       
    tf.keras.layers.Dense(128, activation='relu'),          
    tf.keras.layers.Dense(10, activation='softmax')         
])
epochs = 10
batch_size = 32
learning_rate = 0.005
modelo.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate),
    loss=tf.keras.losses.SparseCategoricalCrossentropy(),
    metrics=['accuracy']
)
# Calcular el número de imágenes en el conjunto de entrenamiento y prueba
# y ajustar el tamaño del lote
num_img_entrenamiento = metadatos.splits['train'].num_examples
num_img_prueba = metadatos.splits['test'].num_examples
datos_entrenamiento = datos_entrenamiento.repeat().shuffle(num_img_entrenamiento).batch(batch_size)
datos_prueba = datos_prueba.batch(batch_size)
steps_per_epoch = math.ceil(num_img_entrenamiento / batch_size)# Entrenamiento
modelo.fit(datos_entrenamiento, epochs=epochs, steps_per_epoch=steps_per_epoch) 
test_loss, test_accuracy = modelo.evaluate(datos_prueba, steps=math.ceil(num_img_prueba / batch_size)) # Evaluar
print(f"Precisión en datos de prueba: {test_accuracy*100:.2f}%")
modelo.save("modelo_digitos.h5") # Guardar el modelo como h5 porque es más ligero que el formato de tensorflow
