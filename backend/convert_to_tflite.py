import tensorflow as tf

# Cargar el modelo
# modelo = tf.keras.models.load_model("numeros.keras")
modelo = tf.keras.models.load_model("C:/Users/OCHOA/Desktop/Gilberto Trabajos/Visión Por Computadora Y Machine Learning/red_neuronal_01/proyecto_digitos/backend/numeros1.keras")

# Convertir a TensorFlow Lite
converter = tf.lite.TFLiteConverter.from_keras_model(modelo)
modelo_tflite = converter.convert()

# Guardar el modelo
with open("numeros1.tflite", "wb") as f:
    f.write(modelo_tflite)

print("Modelo convertido a .tflite correctamente.")
