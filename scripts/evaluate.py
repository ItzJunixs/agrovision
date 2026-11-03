"""Evaluacion final (fase 4): mide cada modelo entrenado en el test y en fotos de campo.

Se ejecuta UNA sola vez, cuando ya terminaron todos los entrenamientos. Por cada corrida mide:
  - test limpio de data/splits.csv (8.834 imagenes que ningun modelo vio)
  - carpeta test de Kaggle (33 imagenes; la clase va en el nombre del archivo)
  - test de PlantDoc (236 fotos de campo tomadas de internet, 27 clases equivalentes)
  - tamano del modelo y tiempo por imagen

Salidas:
  reports/<ID>/test/       metricas, reporte por clase, matriz de confusion y errores en el test
  reports/<ID>/plantdoc/   lo mismo con las fotos de campo
  reports/evaluacion_final.csv / .md / .png   tabla y grafica comparativa

PlantDoc se descarga una vez (solo la carpeta test, unos 70 MB):
  git clone --depth 1 --filter=blob:none --sparse https://github.com/pratikkayal/PlantDoc-Dataset ~/data/plantdoc
  cd ~/data/plantdoc && git sparse-checkout set test

Uso:
  python scripts/evaluate.py --data-root "~/data/New Plant Diseases Dataset(Augmented)/New Plant Diseases Dataset(Augmented)" \
      --kaggle-test ~/data/test/test --plantdoc ~/data/plantdoc/test
"""
import argparse
import json
import re
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import train  # noqa: E402  (registra PreprocesoResNet y reutiliza la evaluacion)

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import tensorflow as tf  # noqa: E402
import keras  # noqa: E402

ORDEN = ["E1", "C1", "C2", "E2", "E3", "E4", "E5"]
CANDIDATOS = ["E1", "E2", "E3", "E4", "E5"]   # C1 y C2 solo son la curva de aprendizaje

KAGGLE_TEST = {
    "AppleCedarRust": "Apple___Cedar_apple_rust", "AppleScab": "Apple___Apple_scab",
    "CornCommonRust": "Corn_(maize)___Common_rust_", "PotatoEarlyBlight": "Potato___Early_blight",
    "PotatoHealthy": "Potato___healthy", "TomatoEarlyBlight": "Tomato___Early_blight",
    "TomatoHealthy": "Tomato___healthy", "TomatoYellowCurlVirus": "Tomato___Tomato_Yellow_Leaf_Curl_Virus",
}
# carpeta de PlantDoc -> clase de PlantVillage
PLANTDOC = {
    "Apple Scab Leaf": "Apple___Apple_scab", "Apple leaf": "Apple___healthy",
    "Apple rust leaf": "Apple___Cedar_apple_rust", "Bell_pepper leaf spot": "Pepper,_bell___Bacterial_spot",
    "Bell_pepper leaf": "Pepper,_bell___healthy", "Blueberry leaf": "Blueberry___healthy",
    "Cherry leaf": "Cherry_(including_sour)___healthy",
    "Corn Gray leaf spot": "Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot",
    "Corn leaf blight": "Corn_(maize)___Northern_Leaf_Blight", "Corn rust leaf": "Corn_(maize)___Common_rust_",
    "Peach leaf": "Peach___healthy", "Potato leaf early blight": "Potato___Early_blight",
    "Potato leaf late blight": "Potato___Late_blight", "Raspberry leaf": "Raspberry___healthy",
    "Soyabean leaf": "Soybean___healthy", "Squash Powdery mildew leaf": "Squash___Powdery_mildew",
    "Strawberry leaf": "Strawberry___healthy", "Tomato Early blight leaf": "Tomato___Early_blight",
    "Tomato Septoria leaf spot": "Tomato___Septoria_leaf_spot", "Tomato leaf bacterial spot": "Tomato___Bacterial_spot",
    "Tomato leaf late blight": "Tomato___Late_blight", "Tomato leaf mosaic virus": "Tomato___Tomato_mosaic_virus",
    "Tomato leaf yellow virus": "Tomato___Tomato_Yellow_Leaf_Curl_Virus", "Tomato leaf": "Tomato___healthy",
    "Tomato mold leaf": "Tomato___Leaf_Mold", "grape leaf black rot": "Grape___Black_rot", "grape leaf": "Grape___healthy",
}


def se_puede_leer(ruta):
    try:
        tf.io.decode_image(tf.io.read_file(ruta), channels=3, expand_animations=False)
        return True
    except Exception:
        return False


def tabla_carpeta(filas, clases):
    df = pd.DataFrame(filas, columns=["archivo", "clase"])
    df = df[[se_puede_leer(a) for a in df["archivo"]]].copy()
    df["y"] = df["clase"].map({c: i for i, c in enumerate(clases)})
    return df.reset_index(drop=True)


def tiempo_por_imagen(nucleo, n=30):
    x = np.random.uniform(0, 255, (1, train.IMG, train.IMG, 3)).astype("float32")
    nucleo(x, training=False)   # calentamiento
    t0 = time.perf_counter()
    for _ in range(n):
        nucleo(x, training=False)
    return (time.perf_counter() - t0) / n * 1000


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-root", required=True)
    ap.add_argument("--splits", default="data/splits.csv")
    ap.add_argument("--kaggle-test", help="carpeta con las 33 imagenes de test de Kaggle")
    ap.add_argument("--plantdoc", help="carpeta test de PlantDoc")
    ap.add_argument("--models", default="models/runs")
    ap.add_argument("--reports", default="reports")
    ap.add_argument("--exps", nargs="*", default=ORDEN)
    ap.add_argument("--sufijo", default="", help="'_smoke' para probar con los modelos de la prueba de humo")
    ap.add_argument("--fraccion", type=float, default=1.0, help="fraccion del test (solo para pruebas)")
    ap.add_argument("--lote", type=int, default=32)
    a = ap.parse_args()

    df = pd.read_csv(a.splits)
    clases = sorted(df["clase"].unique())
    assert set(PLANTDOC.values()) <= set(clases) and set(KAGGLE_TEST.values()) <= set(clases)
    raiz = Path(a.data_root).expanduser()
    test = df[df["conjunto"] == "test"].copy()
    if a.fraccion < 1:
        test = test.groupby("clase").sample(frac=a.fraccion, random_state=42)
    test["archivo"] = [str(raiz / r) for r in test["ruta"]]
    test["y"] = test["clase"].map({c: i for i, c in enumerate(clases)})
    test = test.reset_index(drop=True)

    kaggle = None
    if a.kaggle_test:
        filas = [(str(p), KAGGLE_TEST[re.sub(r"\d+$", "", p.stem)])
                 for p in sorted(Path(a.kaggle_test).expanduser().glob("*")) if re.sub(r"\d+$", "", p.stem) in KAGGLE_TEST]
        kaggle = tabla_carpeta(filas, clases)
    plantdoc = None
    if a.plantdoc:
        filas = [(str(p), PLANTDOC[p.parent.name]) for p in sorted(Path(a.plantdoc).expanduser().glob("*/*"))
                 if p.parent.name in PLANTDOC and p.is_file()]
        plantdoc = tabla_carpeta(filas, clases)
        print(f"PlantDoc: {len(plantdoc)} fotos legibles de {len(filas)}")
    cultivo = np.array([c.split("___")[0] for c in clases])

    resultados = []
    for e in a.exps:
        ruta = Path(a.models) / f"{e}{a.sufijo}" / "model.keras"
        if not ruta.exists():
            print(f"[{e}] sin modelo en {ruta}, se omite")
            continue
        print(f"[{e}] evaluando...")
        nucleo = keras.models.load_model(ruta)
        carpeta = Path(a.reports) / f"{e}{a.sufijo}"
        (carpeta / "test").mkdir(parents=True, exist_ok=True)
        m = train.evaluar(nucleo, test, clases, a.lote, carpeta / "test")
        fila = {"ID": e, "Acc. test": m["accuracy"], "F1 test": m["f1_macro"], "Top-3 test": m["top3_accuracy"],
                "Recall enferma": m["recall_enferma"], "Confianza prom.": m["confianza_promedio"]}
        if kaggle is not None:
            pk = nucleo.predict(train.dataset(kaggle, a.lote, False, 0), verbose=0).argmax(1)
            fila["Acc. Kaggle 33"] = round(float(np.mean(pk == kaggle["y"].values)), 4)
        if plantdoc is not None:
            (carpeta / "plantdoc").mkdir(exist_ok=True)
            mp = train.evaluar(nucleo, plantdoc, clases, a.lote, carpeta / "plantdoc")
            probs = nucleo.predict(train.dataset(plantdoc, a.lote, False, 0), verbose=0)
            acierto_cultivo = float(np.mean(cultivo[probs.argmax(1)] == cultivo[plantdoc["y"].values]))
            mp["accuracy_cultivo"] = round(acierto_cultivo, 4)
            (carpeta / "plantdoc" / "metrics.json").write_text(json.dumps(mp, indent=2), encoding="utf-8")
            fila.update({"Acc. PlantDoc": mp["accuracy"], "Top-3 PlantDoc": mp["top3_accuracy"],
                         "Cultivo PlantDoc": mp["accuracy_cultivo"]})
        fila["MB"] = round(ruta.stat().st_size / 1e6, 1)
        fila["ms/imagen"] = round(tiempo_por_imagen(nucleo), 1)
        (carpeta / "test" / "metrics.json").write_text(json.dumps({**m, **fila}, indent=2), encoding="utf-8")
        resultados.append(fila)
        keras.backend.clear_session()

    if not resultados:
        raise SystemExit("No se encontraron modelos para evaluar.")
    tabla = pd.DataFrame(resultados)
    sal = Path(a.reports)
    tabla.to_csv(sal / f"evaluacion_final{a.sufijo}.csv", index=False)
    pct = [c for c in tabla.columns if c not in ("ID", "MB", "ms/imagen")]
    md = ["| " + " | ".join(tabla.columns) + " |", "|" + " --- |" * len(tabla.columns)]
    for _, r in tabla.iterrows():
        md.append("| " + " | ".join(f"{r[c]:.1%}" if c in pct else str(r[c]) for c in tabla.columns) + " |")
    (sal / f"evaluacion_final{a.sufijo}.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    print("\n".join(md))

    # grafica: test de laboratorio frente a fotos de campo
    cols = [c for c in ("Acc. test", "Acc. PlantDoc") if c in tabla.columns]
    fig, ax = plt.subplots(figsize=(9, 4.5))
    x = np.arange(len(tabla))
    ancho = 0.8 / len(cols)
    for i, (c, color) in enumerate(zip(cols, ["#2f7d32", "#d9822b"])):
        barras = ax.bar(x + (i - (len(cols) - 1) / 2) * ancho, tabla[c] * 100, ancho, color=color,
                        label="Test (laboratorio)" if c == "Acc. test" else "PlantDoc (campo)")
        ax.bar_label(barras, fmt="%.0f", fontsize=8, padding=2)
    ax.axhline(70, color="#b23b2a", ls="--", lw=1)
    ax.text(len(tabla) - 0.5, 71, "meta 70 %", color="#b23b2a", fontsize=8, ha="right")
    ax.set_xticks(x, tabla["ID"])
    ax.set_ylim(0, 105)
    ax.set_ylabel("Accuracy (%)")
    ax.set_title("Evaluación final: fotos de laboratorio frente a fotos de campo")
    ax.legend()
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(sal / f"evaluacion_final{a.sufijo}.png", dpi=130)
    plt.close(fig)

    # ganador: mayor F1 en test; si otro queda a menos de 1 punto y pesa menos, gana el liviano
    cand = tabla[tabla["ID"].isin(CANDIDATOS)].sort_values("F1 test", ascending=False)
    if len(cand):
        mejor = cand.iloc[0]
        cerca = cand[cand["F1 test"] >= mejor["F1 test"] - 0.01].sort_values("MB")
        ganador = cerca.iloc[0]
        print(f"\nGanador sugerido: {ganador['ID']} (F1 test {ganador['F1 test']:.1%}, {ganador['MB']} MB)")
        (sal / f"ganador{a.sufijo}.txt").write_text(str(ganador["ID"]), encoding="utf-8")


if __name__ == "__main__":
    main()
