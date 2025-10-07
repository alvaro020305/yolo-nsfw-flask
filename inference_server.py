import os
import gc
import torch
from fastapi import FastAPI, File, UploadFile, Request
from fastapi.responses import JSONResponse
from ultralytics import YOLO
from PIL import Image
import uvicorn

# --- Configuración ---
NSFW_LABELS = {"penis", "anus", "make_love"}
model = None

app = FastAPI(title="ERA-X Anti NSFW API", version="1.1")


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
@app.post("/check_batch")
async def check_images_batch(request: Request, images: list[UploadFile] = File(...)):
    try:
        model_instance = get_model()
    except Exception as e:
        print(f"❌ Error cargando modelo: {e}")
        return JSONResponse(status_code=500, content={"error": "No se pudo cargar el modelo."})

    try:
        for image_file in images:
            image = Image.open(image_file.file)
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
                        return {"is_nsfw": True, "label": label_name}

        print("✅ Lote procesado. Sin contenido NSFW.")
        return {"is_nsfw": False}

    except Exception as e:
        print(f"⚠️ Error procesando imagen: {e}")
        return JSONResponse(status_code=500, content={"error": str(e)})

    finally:
        torch.cuda.empty_cache()
        gc.collect()


# --- Endpoint de salud ---
@app.get("/ping")
async def ping():
    return {"status": "ok"}


# --- Middleware para cerrar conexiones persistentes (evita 502) ---
@app.middleware("http")
async def add_connection_close_header(request: Request, call_next):
    response = await call_next(request)
    response.headers["Connection"] = "close"
    return response


# --- Inicio del servidor ---
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8080))
    print(f"🚀 Servidor FastAPI corriendo en http://0.0.0.0:{port}")
    uvicorn.run("inference_server:app", host="0.0.0.0", port=port)
