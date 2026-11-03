"""Carga el modelo TFLite de AgroVision y hace predicciones.

El archivo models/agrovision.tflite lo genera scripts/export_tflite.py con el modelo ganador.
Se usa ai-edge-litert (runtime de TensorFlow Lite) en lugar de TensorFlow completo
para que el servicio quepa en los 512 MB de RAM del plan gratuito de Render.
El modelo recibe la imagen RGB de 224x224 con valores 0-255; el preprocesamiento
de MobileNetV2 va dentro del modelo.
"""
import json
from functools import lru_cache

import numpy as np
from django.conf import settings
from PIL import Image

MODEL_PATH = settings.BASE_DIR / "models" / "agrovision.tflite"
LABELS_PATH = settings.BASE_DIR / "models" / "labels.json"

CROPS = {"Apple": "Manzana", "Blueberry": "Arándano", "Cherry_(including_sour)": "Cereza",
         "Corn_(maize)": "Maíz", "Grape": "Uva", "Orange": "Naranja", "Peach": "Durazno",
         "Pepper,_bell": "Pimentón", "Potato": "Papa", "Raspberry": "Frambuesa", "Soybean": "Soya",
         "Squash": "Calabaza", "Strawberry": "Fresa", "Tomato": "Tomate"}
DISEASES = {"Apple_scab": "Sarna", "Black_rot": "Podredumbre negra", "Cedar_apple_rust": "Roya del cedro",
            "healthy": "Sana", "Powdery_mildew": "Oídio",
            "Cercospora_leaf_spot Gray_leaf_spot": "Mancha gris (Cercospora)",
            "Common_rust_": "Roya común", "Northern_Leaf_Blight": "Tizón foliar del norte",
            "Esca_(Black_Measles)": "Yesca (sarampión negro)",
            "Leaf_blight_(Isariopsis_Leaf_Spot)": "Tizón de la hoja (Isariopsis)",
            "Haunglongbing_(Citrus_greening)": "Huanglongbing (enverdecimiento)",
            "Bacterial_spot": "Mancha bacteriana", "Early_blight": "Tizón temprano",
            "Late_blight": "Tizón tardío", "Leaf_scorch": "Quemadura de la hoja", "Leaf_Mold": "Moho de la hoja",
            "Septoria_leaf_spot": "Mancha por Septoria",
            "Spider_mites Two-spotted_spider_mite": "Ácaro de dos manchas", "Target_Spot": "Mancha diana",
            "Tomato_Yellow_Leaf_Curl_Virus": "Virus del rizado amarillo",
            "Tomato_mosaic_virus": "Virus del mosaico"}


def nombre_legible(label: str) -> dict:
    crop, disease = label.split("___")
    return {"clase": label, "cultivo": CROPS.get(crop, crop),
            "estado": DISEASES.get(disease, disease.replace("_", " ")), "sana": disease == "healthy"}


@lru_cache(maxsize=1)
def _modelo():
    """Carga el interprete una sola vez por proceso."""
    from ai_edge_litert.interpreter import Interpreter
    interp = Interpreter(model_path=str(MODEL_PATH))
    interp.allocate_tensors()
    labels = json.loads(LABELS_PATH.read_text(encoding="utf-8"))
    return interp, interp.get_input_details()[0], interp.get_output_details()[0], labels


def predecir(img: Image.Image, k: int = 3) -> list[dict]:
    interp, inp, out, labels = _modelo()
    size = int(inp["shape"][1])
    x = np.asarray(img.convert("RGB").resize((size, size), Image.BILINEAR), dtype=np.float32)[None]
    interp.set_tensor(inp["index"], x)
    interp.invoke()
    probs = interp.get_tensor(out["index"])[0]
    top = np.argsort(-probs)[:k]
    return [{**nombre_legible(labels[i]), "probabilidad": round(float(probs[i]), 4)} for i in top]
