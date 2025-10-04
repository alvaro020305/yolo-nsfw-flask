import os
import io
from flask import Flask, request, jsonify
from ultralytics import YOLO
from PIL import Image

# --------------------------------------------------------------------------
# PASO 1: INICIALIZACIÓN DEL SERVIDOR
# Esto se ejecuta UNA SOLA VEZ cuando inicias este script.
# --------------------------------------------------------------------------

# Se crea la aplicación del servidor web.
app = Flask(__name__)

# Se carga el modelo YOLO en la memoria RAM (y VRAM si hay GPU).
# Esta es la clave de la velocidad: el modelo estará "caliente" y listo para usar.
try:
    # Busca el modelo en el mismo directorio donde está este script.
    script_dir = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(script_dir, "erax-anti-nsfw-yolo11n-v1.1.pt")
    
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"FATAL: El archivo del modelo no se encontró en la ruta esperada: {model_path}")
    
    model = YOLO(model_path)
    print("✅ Modelo YOLO cargado en memoria. Servidor listo en http://127.0.0.1:5000")

except Exception as e:
    print(f"❌ ERROR FATAL: No se pudo cargar el modelo YOLO. El servidor no funcionará. Error: {e}")
    model = None

# --------------------------------------------------------------------------
# PASO 2: EL ENDPOINT DE ANÁLISIS
# Esta función se ejecutará cada vez que Node.js le envíe imágenes.
# --------------------------------------------------------------------------

@app.route('/check_batch', methods=['POST'])
def check_images_batch():
    # Si el modelo no se cargó, devolvemos un error.
    if model is None:
        return jsonify({"error": "El modelo de IA no está disponible. Revisa los logs del servidor."}), 500

    # Verificamos que la petición contenga archivos con el nombre 'images'.
    if 'images' not in request.files:
        return jsonify({"error": "La petición no contiene el campo 'images'."}), 400

    # Obtenemos la lista de todos los archivos enviados.
    image_files = request.files.getlist('images')

    # Procesamos cada imagen recibida.
    for image_file in image_files:
        try:
            # Leemos la imagen directamente desde la memoria, sin guardarla en disco.
            image = Image.open(image_file.stream)

            # Ejecutamos el modelo en la imagen.
            results = model(image, conf=0.2, iou=0.3, verbose=False)

            # Salida anticipada: si UNA SOLA imagen es NSFW, terminamos y respondemos.
            if len(results[0].boxes) > 0:
                print(f"🚨 NSFW detectado en un frame. Respondiendo al bot...")
                return jsonify({'is_nsfw': True})
        except Exception as e:
            # Si un frame está corrupto o da error, lo ignoramos y continuamos.
            print(f"⚠️ Advertencia: Error procesando un frame individual: {e}")
            continue

    # Si el bucle termina, es porque todos los frames eran seguros.
    print("✅ Lote de frames procesado. Resultado: Safe.")
    return jsonify({'is_nsfw': False})

# --------------------------------------------------------------------------
# PASO 3: PONER EN MARCHA EL SERVIDOR
# --------------------------------------------------------------------------
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 8080)))
