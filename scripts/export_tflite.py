"""Fase 5: exporta el modelo ganador a TFLite y actualiza los archivos que usa la app.

1. Reconstruye el modelo en float32 (los modelos se entrenan con precision mixta) y copia sus pesos.
2. Lo convierte a TFLite con cuantizacion dinamica (unas 4 veces mas liviano).
3. Comprueba en una muestra del test que el TFLite no pierda mas de 1 punto de accuracy.
4. Escribe models/agrovision.tflite, models/labels.json y reports/app/ (metricas, historial y
   matriz de confusion que muestra la pagina).

Uso (despues de scripts/evaluate.py, que deja el ganador en reports/ganador.txt):
  python scripts/export_tflite.py --data-root "~/data/New Plant Diseases Dataset(Augmented)/New Plant Diseases Dataset(Augmented)"
  python scripts/export_tflite.py --exp E3 --data-root ...     # para exportar otro
"""
import argparse
import json
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import train  # noqa: E402

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import tensorflow as tf  # noqa: E402
import keras  # noqa: E402

NOMBRES = {
    "E1": "CNN propia entrenada desde cero",
    "E2": "ResNet50 con la base congelada",
    "E3": "ResNet50 con fine-tuning (Adam)",
    "E4": "ResNet50 con fine-tuning (SGD con momentum)",
    "E5": "MobileNetV2 con fine-tuning",
    "C1": "ResNet50 congelada, 100 imágenes por clase",
    "C2": "ResNet50 congelada, 500 imágenes por clase",
}


def leer_json(ruta):
    return json.loads(Path(ruta).read_text(encoding="utf-8")) if Path(ruta).exists() else None


def historial(rep, exp):
    """Historial por epoca; si la corrida parte de otra (fine-tuning), se antepone el de la original."""
    cfg = leer_json(rep / exp / "config.json") or {}
    partes = []
    if cfg.get("desde") and (rep / cfg["desde"] / "history.csv").exists():
        partes.append(pd.read_csv(rep / cfg["desde"] / "history.csv"))
    partes.append(pd.read_csv(rep / exp / "history.csv"))
    h = pd.concat(partes, ignore_index=True)
    return {k: [round(float(v), 5) for v in h[k]] for k in ("loss", "val_loss", "accuracy", "val_accuracy")}


def graficar_confusion(cm, clases, titulo, salida):
    cmn = cm / np.maximum(cm.sum(1, keepdims=True), 1)
    etiquetas = [c.replace("___", " · ").replace("_", " ")[:34] for c in clases]
    fig, ax = plt.subplots(figsize=(14, 12))
    im = ax.imshow(cmn, cmap="Greens", vmin=0, vmax=1)
    ax.set_xticks(range(len(clases)), etiquetas, rotation=90, fontsize=7)
    ax.set_yticks(range(len(clases)), etiquetas, fontsize=7)
    ax.set_xlabel("Predicción")
    ax.set_ylabel("Clase real")
    ax.set_title(titulo)
    fig.colorbar(im, fraction=0.03)
    fig.tight_layout()
    fig.savefig(salida, dpi=130)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--exp", help="corrida a exportar (por defecto, la de reports/ganador.txt)")
    ap.add_argument("--data-root", required=True)
    ap.add_argument("--splits", default="data/splits.csv")
    ap.add_argument("--models", default="models/runs")
    ap.add_argument("--reports", default="reports")
    ap.add_argument("--out", default="models/agrovision.tflite")
    ap.add_argument("--app-dir", default="reports/app")
    ap.add_argument("--muestras", type=int, default=760, help="imágenes del test para comparar Keras y TFLite")
    ap.add_argument("--version", default="v1.0.0")
    a = ap.parse_args()

    rep = Path(a.reports)
    exp = a.exp or (rep / "ganador.txt").read_text(encoding="utf-8").strip()
    cfg = leer_json(rep / exp / "config.json")
    origen = keras.models.load_model(Path(a.models) / exp / "model.keras")
    clases = json.loads((Path(a.models) / exp / "labels.json").read_text(encoding="utf-8"))

    # 1. copia en float32 de la misma arquitectura
    keras.mixed_precision.set_global_policy("float32")
    modelo = train.construir_nucleo(cfg["arq"], len(clases), pesos=None)
    modelo.set_weights(origen.get_weights())

    # 2. conversion
    conv = tf.lite.TFLiteConverter.from_keras_model(modelo)
    conv.optimizations = [tf.lite.Optimize.DEFAULT]
    tflite = conv.convert()
    salida = Path(a.out)
    salida.parent.mkdir(parents=True, exist_ok=True)
    salida.write_bytes(tflite)
    mb = round(salida.stat().st_size / 1e6, 2)
    print(f"[{exp}] TFLite: {salida} ({mb} MB)")

    # 3. verificacion en una muestra estratificada del test
    df = pd.read_csv(a.splits)
    test = df[df["conjunto"] == "test"]
    n_por_clase = max(1, a.muestras // len(clases))
    test = test.groupby("clase").sample(n=n_por_clase, random_state=42, replace=False).reset_index(drop=True)
    test["archivo"] = [str(Path(a.data_root).expanduser() / r) for r in test["ruta"]]
    test["y"] = test["clase"].map({c: i for i, c in enumerate(clases)})
    x = np.stack([tf.image.resize(tf.io.decode_image(tf.io.read_file(f), channels=3, expand_animations=False),
                                  (train.IMG, train.IMG)).numpy() for f in test["archivo"]]).astype("float32")
    p_keras = modelo.predict(x, batch_size=32, verbose=0).argmax(1)
    interp = tf.lite.Interpreter(model_content=tflite)
    interp.allocate_tensors()
    ent, sal = interp.get_input_details()[0], interp.get_output_details()[0]
    p_tflite = []
    for img in x:
        interp.set_tensor(ent["index"], img[None])
        interp.invoke()
        p_tflite.append(int(interp.get_tensor(sal["index"])[0].argmax()))
    y = test["y"].values
    acc_k, acc_t = float(np.mean(p_keras == y)), float(np.mean(np.array(p_tflite) == y))
    coincide = float(np.mean(p_keras == np.array(p_tflite)))
    print(f"[{exp}] muestra de {len(y)}: Keras {acc_k:.1%} | TFLite {acc_t:.1%} | predicciones iguales {coincide:.1%}")
    if acc_k - acc_t > 0.01:
        raise SystemExit("El TFLite pierde más de 1 punto: revisar antes de desplegar (no se actualizó la app).")

    # 4. archivos de la app
    shutil.copy(Path(a.models) / exp / "labels.json", salida.parent / "labels.json")
    app = Path(a.app_dir)
    app.mkdir(parents=True, exist_ok=True)
    m_test = leer_json(rep / exp / "test" / "metrics.json")
    m_val = leer_json(rep / exp / "metrics.json")
    m_pd = leer_json(rep / exp / "plantdoc" / "metrics.json")
    m = m_test or m_val
    metricas = {
        "version": a.version, "modelo": NOMBRES.get(exp, exp), "experimento": exp, "clases": len(clases),
        "imagenes_train": m_val["imagenes_train"], "conjunto": "prueba" if m_test else "validación",
        "accuracy": m["accuracy"] if not m_test else m_test["Acc. test"],
        "top3_accuracy": m["top3_accuracy"], "f1_macro": m["f1_macro"],
        "plantdoc_accuracy": m_pd["accuracy"] if m_pd else None,
        "tamano_tflite_mb": mb,
        "verificacion_tflite": {"muestras": int(len(y)), "accuracy_keras": round(acc_k, 4), "accuracy_tflite": round(acc_t, 4)},
    }
    (app / "metrics.json").write_text(json.dumps(metricas, indent=2, ensure_ascii=False), encoding="utf-8")
    (app / "history.json").write_text(json.dumps(historial(rep, exp)), encoding="utf-8")
    cm_ruta = rep / exp / ("test" if m_test else "") / "confusion_matrix.npy"
    graficar_confusion(np.load(cm_ruta), clases,
                       f"Matriz de confusión normalizada ({metricas['conjunto']}): {NOMBRES.get(exp, exp)}",
                       app / "confusion_matrix.png")
    print(f"[{exp}] App actualizada en {app}/ y {salida}. Pruébala con: python manage.py runserver")


if __name__ == "__main__":
    main()
