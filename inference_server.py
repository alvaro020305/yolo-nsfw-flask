import os
from flask import Flask, request, jsonify
from ultralytics import YOLO
from PIL import Image

NSFW_LABELS = {"penis", "anus", "make_love"}

app = Flask(__name__)

try:
    script_dir = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(script_dir, "erax-anti-nsfw-yolo11n-v1.1.pt")

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"FATAL: El archivo del modelo no se encontró: {model_path}")

    model = YOLO(model_path)
    print("✅ Modelo YOLO cargado. Servidor listo en http://127.0.0.1:8080")

except Exception as e:
    print(f"❌ ERROR: No se pudo cargar el modelo YOLO. Error: {e}")
    model = None

@app.route('/check_batch', methods=['POST'])
def check_images_batch():
    if model is None:
        return jsonify({"error": "El modelo no está disponible."}), 500

    if 'images' not in request.files:
        return jsonify({"error": "No se recibió campo 'images'."}), 400

    image_files = request.files.getlist('images')

    for image_file in image_files:
        try:
            image = Image.open(image_file.stream)
            results = model(image, conf=0.2, iou=0.3, verbose=False)

            if len(results) > 0 and len(results[0].boxes) > 0:
                names = model.names  # Obtiene el diccionario id→label
                detections = results[0].boxes

                for cls_id in detections.cls:
                    label_name = names[int(cls_id)]
                    if label_name in NSFW_LABELS:
                        print(f"🚨 NSFW detectado: {label_name}")
                        return jsonify({'is_nsfw': True, 'label': label_name})

        except Exception as e:
            print(f"⚠️ Error procesando imagen: {e}")
            continue

    print("✅ Lote procesado. Sin contenido NSFW.")
    return jsonify({'is_nsfw': False})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 8080)))
