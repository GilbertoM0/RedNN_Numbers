# uvicorn app:app --reload                          # COMANDO PARA Ejecutar el servidor FastAPI
#Recuerda el cd backend Gilberto
import tensorflow as tf
import numpy as np
from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
import io

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

interpreter = tf.lite.Interpreter(model_path="numeros.tflite")
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

@app.post("/predecir/")
async def predecir_numero(file: UploadFile = File(...)):
    contenido = await file.read()
    imagen = Image.open(io.BytesIO(contenido)).convert("L")
    imagen = imagen.resize((28, 28))
    # imagen = Image.eval(imagen, lambda x: 255 - x)  # <-- ESTA LÍNEA es crucial
    # 💾 Guardar imagen para depuración
    imagen.save("debug.png")

    imagen_np = np.array(imagen).astype("float32") / 255.0
    imagen_np = imagen_np.reshape(1, 28, 28, 1)

    interpreter.set_tensor(input_details[0]['index'], imagen_np)
    interpreter.invoke()
    output_data = interpreter.get_tensor(output_details[0]['index'])

    clase_predicha = int(np.argmax(output_data))
    return {"digito": clase_predicha}
