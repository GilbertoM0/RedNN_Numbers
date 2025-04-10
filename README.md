# DOCUMENTACION
> [!Important]
> Levante un servidor local para poder realizar las pruebas, el link es el deploy de la pagina pero recomiendo hacerlo localmente todo, ya que no tengo el servidor en la nube aún.

> [!NOTE]
> He completado el entrenamiento de un modelo para clasificar imágenes del dataset MNIST, que descargué utilizando la librería TensorFlow Datasets (tfds). El modelo se entrenó con el objetivo de alcanzar una precisión mínima del 97%, lo cual se validó utilizando el método evaluate() de Keras.

Para poner este modelo a disposición a través de una aplicación web, opté por utilizar FastAPI, un framework ligero y rápido que permite crear APIs de alto rendimiento. El servidor está configurado para recibir solicitudes de clasificación y devolver las predicciones correspondientes.

Este enfoque proporciona una interfaz sencilla y eficiente para interactuar con el modelo entrenado, optimizando tanto el rendimiento como la facilidad de despliegue en producción.

# Responda las siguientes preguntas:
## Que pasa si quita neuronas a la cpa oculta?
R: Lo note mas rapido al momento de entrenar
## Qué pasa si agrega más de 3 capas ocultas con 60 neuronas?
R: El entrenamiento lo vi mas lento con el mismo numero de iteraciones (10) y en porcentajes bastante similares eso si.
![60neuronas](https://github.com/user-attachments/assets/959b4df4-8fe3-45b6-896a-999e170b6b21)

## Qué pasa con una capa oculta de 128 neuronas?
R: Note que hizo un entrenamiento mas rapido que al ser 3 capas ocultas.
Tambien se muestra el accuracy con la formula que fue del 97.6% (esta hasta abajo de la terminal).
![128neuronas](https://github.com/user-attachments/assets/ccc17dcc-a3a4-4f97-aa33-bc634d173da0)
