import os
import gc
import torch
from flask import Flask, request, jsonify
from ultralytics import YOLO
from PIL import Image

# --- Configuración ---
NSFW_LABELS = {"penis", "anus", "make_love"}
model = None

app = Flask(__name__)


# --- Lazy load del modelo ---
def get_model():
    global model
    if model is None:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        model_path = os.path.join(script_dir, "erax-anti-nsfw-yolo11n-v1.1.pt")

        if not os.path.exists(model_path):
            raise FileNotFoundError(f"❌ Modelo no encontrado: {model_path}")

        model = YOLO(model_path)
        print("✅ Modelo YOLO cargado correctamente.")
    return model


# --- Endpoint principal ---
@app.route("/check_batch", methods=["POST"])
def check_images_batch():
    try:
        model_instance = get_model()
    except Exception as e:
        print(f"❌ Error cargando modelo: {e}")
        return jsonify({"error": "No se pudo cargar el modelo."}), 500

    if "images" not in request.files:
        return jsonify({"error": "No se recibió el campo 'images'."}), 400

    image_files = request.files.getlist("images")

    try:
        for image_file in image_files:
            image = Image.open(image_file.stream).convert("RGB")
            results = model_instance(image, conf=0.2, iou=0.3, verbose=False)

            if len(results) > 0 and len(results[0].boxes) > 0:
                names = model_instance.names
                detections = results[0].boxes

                for cls_id in detections.cls:
                    label_name = names[int(cls_id)]
                    if label_name in NSFW_LABELS:
                        print(f"🚨 NSFW detectado: {label_name}")
                        torch.cuda.empty_cache()
                        gc.collect()
                        return jsonify({"is_nsfw": True, "label": label_name})

        print("✅ Lote procesado. Sin contenido NSFW.")
        return jsonify({"is_nsfw": False})

    except Exception as e:
        print(f"⚠️ Error procesando imagen: {e}")
        return jsonify({"error": str(e)}), 500

    finally:
        torch.cuda.empty_cache()
        gc.collect()


# --- Endpoint de salud ---
@app.route("/ping", methods=["GET"])
def ping():
    return jsonify({"status": "ok"})


# --- Run del servidor ---
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8080))
    print(f"🚀 Servidor Flask corriendo en http://0.0.0.0:{port}")
    app.run(host="0.0.0.0", port=port)
