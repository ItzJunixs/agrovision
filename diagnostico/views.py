import base64
import io
import json

from django.conf import settings
from django.http import JsonResponse
from django.shortcuts import render
from PIL import Image, UnidentifiedImageError

from .predictor import MODEL_PATH, predecir

REPORTS = settings.BASE_DIR / "reports" / "app"   # lo actualiza scripts/export_tflite.py
UMBRAL_CONFIANZA = 0.5   # por debajo, la app pide otra foto en vez de dar un diagnostico seguro


def _leer_json(nombre):
    return json.loads((REPORTS / nombre).read_text(encoding="utf-8"))


def _porcentaje(valor):
    """Un decimal; dos si con uno se veria 100 % sin serlo (99,97 % no es 100 %)."""
    txt = f"{valor * 100:.1f}"
    if txt == "100.0" and valor < 1:
        txt = f"{valor * 100:.2f}"
    return txt.replace(".", ",") + " %"


def index(request):
    metricas = _leer_json("metrics.json")
    ctx = {
        "metricas": metricas,
        # porcentajes con un decimal: 99,95 % no debe verse como 100 %
        "pct": {k: _porcentaje(metricas[k])
                for k in ("accuracy", "top3_accuracy", "f1_macro", "plantdoc_accuracy") if metricas.get(k) is not None},
        # grafica 1: curvas de entrenamiento (perdida y accuracy por epoca)
        "historial": _leer_json("history.json"),
    }
    if request.method == "POST":
        archivo = request.FILES.get("imagen")
        if not archivo:
            ctx["error"] = "Selecciona una imagen."
        else:
            datos = archivo.read()
            try:
                img = Image.open(io.BytesIO(datos))
                img.load()
            except (UnidentifiedImageError, OSError):
                ctx["error"] = "El archivo no es una imagen válida."
            else:
                top3 = predecir(img)
                ctx["resultado"] = top3[0]
                ctx["confianza_baja"] = top3[0]["probabilidad"] < UMBRAL_CONFIANZA
                # grafica 3: probabilidades del top-3 de la foto subida
                ctx["top3"] = top3
                vista = img.convert("RGB")
                vista.thumbnail((480, 480))
                buf = io.BytesIO()
                vista.save(buf, "JPEG", quality=85)
                ctx["imagen_b64"] = base64.b64encode(buf.getvalue()).decode()
    return render(request, "diagnostico/index.html", ctx)


def health(request):
    return JsonResponse({"status": "ok", "modelo": MODEL_PATH.name})
