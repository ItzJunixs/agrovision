"""Modelo base de AgroVision: MobileNetV2 preentrenado en ImageNet como extractor
congelado + cabeza densa entrenada sobre las 38 clases de PlantVillage.

Etapas:
  1. Carga train/valid desde data/subset (carpetas por clase).
  2. Extrae embeddings de 1280 dimensiones con MobileNetV2 congelado (una sola pasada).
  3. Entrena la cabeza (Dropout + Dense softmax) sobre los embeddings.
  4. Evalua en valid y en la carpeta test original (33 imagenes, etiqueta en el nombre).
  5. Guarda el modelo completo (.keras), una version TFLite para despliegue y las metricas.

Uso:
  python scripts/train_baseline.py --data data/subset --weights models/mobilenet_v2_no_top.h5
"""
import argparse, json, re, time, urllib.request
from pathlib import Path

import numpy as np
import tensorflow as tf
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score

# nombre de archivo en test/ -> clase del dataset
TEST_MAP = {
    "AppleCedarRust": "Apple___Cedar_apple_rust",
    "AppleScab": "Apple___Apple_scab",
    "CornCommonRust": "Corn_(maize)___Common_rust_",
    "PotatoEarlyBlight": "Potato___Early_blight",
    "PotatoHealthy": "Potato___healthy",
    "TomatoEarlyBlight": "Tomato___Early_blight",
    "TomatoHealthy": "Tomato___healthy",
    "TomatoYellowCurlVirus": "Tomato___Tomato_Yellow_Leaf_Curl_Virus",
}


WEIGHTS_URL = ("https://github.com/JonathanCMitchell/mobilenet_v2_keras/releases/download/v1.1/"
               "mobilenet_v2_weights_tf_dim_ordering_tf_kernels_1.0_224_no_top.h5")


def build_base(img, weights):
    if weights != "imagenet" and not Path(weights).exists():
        print("Descargando pesos de MobileNetV2 (ImageNet)...")
        Path(weights).parent.mkdir(parents=True, exist_ok=True)
        urllib.request.urlretrieve(WEIGHTS_URL, weights)
    base = tf.keras.applications.MobileNetV2(
        input_shape=(img, img, 3), include_top=False, weights=weights, pooling="avg")
    base.trainable = False
    return base


def embed_split(base, split_dir, classes, cache_dir, img, deadline):
    """Calcula los embeddings clase por clase y los guarda en cache (.npy).
    Si se acaba el tiempo (deadline) se detiene; al volver a ejecutar continua donde quedo."""
    cache_dir.mkdir(parents=True, exist_ok=True)
    for i, cls in enumerate(classes):
        f = cache_dir / f"{i:02d}.npy"
        if f.exists():
            continue
        if time.time() > deadline:
            return False
        files = sorted(p for p in (split_dir / cls).iterdir() if p.is_file())
        batch = np.stack([np.array(tf.keras.utils.load_img(p, target_size=(img, img)), dtype="float32") for p in files])
        x = tf.keras.applications.mobilenet_v2.preprocess_input(batch)
        np.save(f, base.predict(x, batch_size=32, verbose=0))
    return True


def load_cache(cache_dir, n_classes):
    X, y = [], []
    for i in range(n_classes):
        e = np.load(cache_dir / f"{i:02d}.npy")
        X.append(e); y.append(np.full(len(e), i))
    return np.concatenate(X), np.concatenate(y)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/subset")
    ap.add_argument("--weights", default="models/mobilenet_v2_no_top.h5")
    ap.add_argument("--img", type=int, default=224)
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--out", default="models")
    ap.add_argument("--reports", default="reports")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--budget", type=int, default=10**9, help="segundos maximos para calcular embeddings en esta ejecucion")
    a = ap.parse_args()
    tf.keras.utils.set_random_seed(a.seed)
    data, out, rep = Path(a.data), Path(a.out), Path(a.reports)
    out.mkdir(exist_ok=True); rep.mkdir(exist_ok=True)

    classes = sorted(p.name for p in (data / "train").iterdir() if p.is_dir())
    base = build_base(a.img, a.weights)
    deadline = time.time() + a.budget
    t0 = time.time()
    done = all(embed_split(base, data / sp, classes, data / "cache" / sp, a.img, deadline) for sp in ("train", "valid"))
    if not done:
        print("EMBEDDINGS INCOMPLETOS: vuelve a ejecutar el script para continuar")
        return
    Xtr, ytr = load_cache(data / "cache" / "train", len(classes))
    Xva, yva = load_cache(data / "cache" / "valid", len(classes))
    t_embed = time.time() - t0
    print(f"Embeddings: train {Xtr.shape}, valid {Xva.shape}")
    perm = np.random.default_rng(a.seed).permutation(len(ytr))
    Xtr, ytr = Xtr[perm], ytr[perm]

    head = tf.keras.Sequential([
        tf.keras.layers.Input((Xtr.shape[1],)),
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.Dense(len(classes), activation="softmax"),
    ])
    head.compile(optimizer=tf.keras.optimizers.Adam(1e-3),
                 loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    hist = head.fit(Xtr, ytr, validation_data=(Xva, yva), epochs=a.epochs, batch_size=64, verbose=2,
                    callbacks=[tf.keras.callbacks.EarlyStopping(patience=6, restore_best_weights=True)])

    # evaluacion en valid
    pva = head.predict(Xva, verbose=0)
    yhat = pva.argmax(1)
    top3 = np.mean([y in np.argsort(-p)[:3] for p, y in zip(pva, yva)])
    report = classification_report(yva, yhat, target_names=classes, output_dict=True, zero_division=0)

    # modelo completo: imagen RGB 0-255 -> probabilidades
    inp = tf.keras.Input((a.img, a.img, 3))
    x = tf.keras.applications.mobilenet_v2.preprocess_input(inp)
    model = tf.keras.Model(inp, head(base(x, training=False)), name="agrovision_baseline")
    model.save(out / "agrovision_baseline.keras")
    tflite = tf.lite.TFLiteConverter.from_keras_model(model)
    tflite.optimizations = [tf.lite.Optimize.DEFAULT]  # cuantizacion dinamica: ~4x mas liviano
    (out / "agrovision_baseline.tflite").write_bytes(tflite.convert())
    (out / "labels.json").write_text(json.dumps(classes, indent=1), encoding="utf-8")

    # test original de Kaggle (33 imagenes)
    test_rows = []
    for f in sorted((data / "test").glob("*")):
        key = re.sub(r"\d+$", "", f.stem)
        img = tf.keras.utils.load_img(f, target_size=(a.img, a.img))
        p = model.predict(np.array(img)[None].astype("float32"), verbose=0)[0]
        pred = classes[int(p.argmax())]
        test_rows.append({"archivo": f.name, "real": TEST_MAP.get(key), "prediccion": pred,
                          "confianza": round(float(p.max()), 4), "acierto": pred == TEST_MAP.get(key)})
    test_acc = float(np.mean([r["acierto"] for r in test_rows])) if test_rows else None

    metrics = {
        "modelo": "MobileNetV2 congelado (ImageNet) + Dense softmax",
        "imagenes_train": int(len(ytr)), "imagenes_valid": int(len(yva)), "clases": len(classes),
        "epocas_entrenadas": len(hist.history["loss"]),
        "valid_accuracy": round(float(accuracy_score(yva, yhat)), 4),
        "valid_f1_macro": round(float(f1_score(yva, yhat, average="macro")), 4),
        "valid_top3_accuracy": round(float(top3), 4),
        "test_kaggle_accuracy": round(test_acc, 4) if test_acc is not None else None,
        "tamano_tflite_mb": round((out / "agrovision_baseline.tflite").stat().st_size / 1e6, 2),
    }
    (rep / "metrics.json").write_text(json.dumps(metrics, indent=2, ensure_ascii=False), encoding="utf-8")
    (rep / "classification_report.json").write_text(json.dumps(report, indent=1), encoding="utf-8")
    (rep / "test_predictions.json").write_text(json.dumps(test_rows, indent=1, ensure_ascii=False), encoding="utf-8")
    (rep / "history.json").write_text(json.dumps({k: [float(v) for v in vs] for k, vs in hist.history.items()}), encoding="utf-8")
    np.save(rep / "confusion_matrix.npy", confusion_matrix(yva, yhat))
    print(json.dumps(metrics, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
