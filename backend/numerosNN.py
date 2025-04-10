# Importar librerias
import tensorflow as tf
import tensorflow_datasets as tfds
import numpy as np
import math
import matplotlib.pyplot as plt

# Descargar el dataset'fashion_mnist'
datos, metadatos = tfds.load('mnist', as_supervised=True, with_info=True)
# Separar los datos de entrenamiento y de prueba
datos_entrenamiento = datos['train']
datos_prueba = datos['test']# Obtener el nombre de las clases
nombre_clases = metadatos.features['label'].names


# Normalizar las imagenes para que los valores de los pixeles
# sean entre 0 y 1, actualmente estan entre 0 y 255
def normalizar(imagenes,etiquetas):
    # Convertir de enteros a flotantes
    imagenes = tf.cast(imagenes, tf.float32)
    # Dividir entre 255 para normalizar la imagen
    imagenes = imagenes / 255
    return imagenes, etiquetas

# Normalizar los datos de entrenamiento y los datos de prueba
datos_entrenamiento = datos_entrenamiento.map(normalizar)
datos_prueba = datos_prueba.map(normalizar)

# Agregar los datos a cache
datos_entrenamiento = datos_entrenamiento.cache()
datos_prueba = datos_prueba.cache()

# Crear modelo
modelo = tf.keras.Sequential([
    # Definir la primera capa de entrada de tipo Flatten
    tf.keras.layers.Flatten(input_shape=(28, 28, 1)),  # ← Se corrige aquí

    tf.keras.layers.Dense(128, activation='relu'),  # ← Definir la primera capa oculta

    tf.keras.layers.Dense(10, activation='softmax')  # ← # Capa de salida. No necesita coma al final
])

# Definir los hiperparametros
epochs = 10
learning_rate = 0.0005
# Definir el tamaño del lote batch size
batch_size = 32

# Compilar el modelo
modelo.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate),
    loss = tf.keras.losses.SparseCategoricalCrossentropy(), # Entropia cruzada categorica dispersa
    metrics = ['accuracy']
)

# Obtener el numero de imagenes de entrenamiento y de prueba
num_img_entrenamiento = metadatos.splits['train'].num_examples
num_img_prueba = metadatos.splits['test'].num_examples


# Aplicar una estrategia para que el entrenamiento aprenda mas rapido y eficiente
# Y mejore la taza de aprendizaje
datos_entrenamiento = datos_entrenamiento.repeat().shuffle(num_img_entrenamiento).batch(batch_size)
datos_prueba = datos_prueba.batch(batch_size)

# Entrenar el modelo
steps_per_epoch = math.ceil(num_img_entrenamiento/batch_size)
historial = modelo.fit(datos_entrenamiento, epochs= epochs, steps_per_epoch = steps_per_epoch)

# Predecir todas las imagenes de prueba
for imagenes_prueba,etiquetas_prueba in datos_prueba.take(1):
    imagenes_prueba = imagenes_prueba.numpy()
    etiquetas_prueba = etiquetas_prueba.numpy()
    predicciones = modelo.predict(imagenes_prueba)

# Predecir nuestra propia imagen
index_imagen = 0
imagen = imagenes_prueba[index_imagen]
plt.imshow(np.reshape(imagen, (28,28)), cmap=plt.cm.binary)
plt.show()

# Realizar prediccion
imagen = np.array([imagen])
p = modelo.predict(imagen)
print(f"La prediccion es: {nombre_clases[np.argmax(p)]}")

# Guardar el modelo en formato SavedModel
# modelo.save("numeros1.keras")


test_loss,test_accuracy = modelo.evaluate(datos_prueba, steps=math.ceil(num_img_prueba/32))
print(f"Con la formula el accuracy es: {test_accuracy}")
print("Modelo guardado correctamente.")