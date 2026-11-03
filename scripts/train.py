"""Entrena y evalua los experimentos del roadmap de AgroVision.

Experimentos (todos con las 38 clases y la division de data/splits.csv):
  E1  CNN propia desde cero, Adam 1e-3
  C1  ResNet50 con la base congelada, 100 imagenes por clase   (curva de aprendizaje)
  C2  ResNet50 con la base congelada, 500 imagenes por clase   (curva de aprendizaje)
  E2  ResNet50 con la base congelada, todas las imagenes
  E3  ResNet50 fine-tuning del bloque conv5 desde E2, Adam 1e-4
  E4  ResNet50 fine-tuning del bloque conv5 desde E2, SGD con momentum 0.9 y tasa 1e-3
  E5  MobileNetV2: base congelada y luego fine-tuning desde la capa 100

Cada corrida guarda:
  reports/<ID>/  config.json, history.csv, metrics.json, classification_report.json,
                 confusion_matrix.npy, training_curves.png, errores.png
  models/runs/<ID>/model.keras   modelo de inferencia (imagen RGB 0-255 -> 38 probabilidades)

Uso:
  python scripts/train.py --exp E1 --data-root "~/data/New Plant Diseases Dataset(Augmented)/New Plant Diseases Dataset(Augmented)"
  python scripts/train.py --exp E2 --data-root ... --smoke      # prueba de humo: 1 epoca con el 5 % de los datos
"""
import argparse
import json
import os
import platform
import time
from pathlib import Path

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker
import numpy as np
import pandas as pd
import tensorflow as tf
import keras
from keras import layers
from sklearn.metrics import (accuracy_score, classification_report, confusion_matrix,
                             f1_score, precision_score, recall_score)

IMG = 224

# ---------------------------------------------------------------- configuracion de experimentos
EXPERIMENTOS = {
    "E1": dict(arq="cnn", etapas=[dict(congelar="nada", opt="adam", lr=1e-3, epocas=30)], lote=64),
    "C1": dict(arq="resnet50", por_clase=100, etapas=[dict(congelar="base", opt="adam", lr=1e-3, epocas=20)], lote=32),
    "C2": dict(arq="resnet50", por_clase=500, etapas=[dict(congelar="base", opt="adam", lr=1e-3, epocas=20)], lote=32),
    "E2": dict(arq="resnet50", etapas=[dict(congelar="base", opt="adam", lr=1e-3, epocas=5)], lote=32),
    "E3": dict(arq="resnet50", desde="E2", etapas=[dict(congelar="hasta:conv5_block1_1_conv", opt="adam", lr=1e-4, epocas=10)], lote=32),
    "E4": dict(arq="resnet50", desde="E2", etapas=[dict(congelar="hasta:conv5_block1_1_conv", opt="sgd", lr=1e-3, epocas=10)], lote=32),
    "E5": dict(arq="mobilenetv2", etapas=[dict(congelar="base", opt="adam", lr=1e-3, epocas=5),
                                          dict(congelar="hasta:100", opt="adam", lr=1e-4, epocas=10)], lote=32),
}


# ---------------------------------------------------------------- capas propias
@keras.saving.register_keras_serializable(package="agrovision")
class PreprocesoResNet(layers.Layer):
    """Preprocesamiento de ResNet50 (estilo caffe): RGB -> BGR y resta la media de ImageNet."""

    def call(self, x):
        x = keras.ops.flip(x, axis=-1)
        return x - keras.ops.convert_to_tensor([103.939, 116.779, 123.68], dtype=x.dtype)


def capa_aumento(seed):
    """Aumento de datos de la semana 3; solo actua durante el entrenamiento."""
    return keras.Sequential([
        layers.RandomFlip("horizontal_and_vertical", seed=seed),
        layers.RandomRotation(15 / 360, fill_mode="reflect", seed=seed),       # +-15 grados
        layers.RandomZoom((-0.2, 0.2), fill_mode="reflect", seed=seed),        # 80-120 %
        layers.RandomTranslation(0.1, 0.1, fill_mode="reflect", seed=seed),    # +-10 %
        layers.RandomBrightness(0.1, value_range=(0, 255), seed=seed),          # +-10 %
        layers.RandomContrast(0.1, seed=seed),
        # ruido gaussiano leve: Keras lo pide en escala 0-1, asi que se escala, se aplica y se regresa a 0-255
        layers.Rescaling(1 / 255),
        layers.GaussianNoise(0.02, seed=seed),
        layers.Rescaling(255),
    ], name="aumento")


def cabeza(x, n_clases, l2=1e-4, dropout=0.3):
    x = layers.GlobalAveragePooling2D(name="gap")(x)
    x = layers.Dropout(dropout, name="dropout")(x)
    x = layers.Dense(n_clases, kernel_regularizer=keras.regularizers.l2(l2), name="logits")(x)
    # la salida queda en float32 aunque se entrene con precision mixta
    return layers.Activation("softmax", dtype="float32", name="probabilidades")(x)


def construir_nucleo(arq, n_clases, pesos):
    """Modelo de inferencia: imagen RGB 0-255 de 224x224 -> probabilidades. No incluye el aumento."""
    inp = keras.Input((IMG, IMG, 3), name="imagen")
    if arq == "cnn":
        x = layers.Rescaling(1 / 255, name="escala")(inp)
        for i, f in enumerate([32, 64, 128, 256], start=1):
            x = layers.Conv2D(f, 3, padding="same", use_bias=False, name=f"conv{i}")(x)
            x = layers.BatchNormalization(name=f"bn{i}")(x)
            x = layers.ReLU(name=f"relu{i}")(x)
            x = layers.MaxPooling2D(2, name=f"pool{i}")(x)
        x = layers.GlobalAveragePooling2D(name="gap")(x)
        x = layers.Dense(128, activation="relu", name="densa")(x)
        x = layers.Dropout(0.3, name="dropout")(x)
        x = layers.Dense(n_clases, name="logits")(x)
        out = layers.Activation("softmax", dtype="float32", name="probabilidades")(x)
        return keras.Model(inp, out, name="cnn_propia")
    if arq == "resnet50":
        x = PreprocesoResNet(name="preproceso")(inp)
        base = keras.applications.ResNet50(include_top=False, weights=pesos, input_shape=(IMG, IMG, 3))
    elif arq == "mobilenetv2":
        x = layers.Rescaling(1 / 127.5, offset=-1, name="preproceso")(inp)   # rango -1 a 1
        base = keras.applications.MobileNetV2(include_top=False, weights=pesos, input_shape=(IMG, IMG, 3))
    else:
        raise ValueError(arq)
    x = base(x, training=False)   # BatchNorm de la base siempre en modo inferencia
    return keras.Model(inp, cabeza(x, n_clases), name=arq)


def aplicar_congelado(nucleo, regla):
    """regla: 'nada' | 'base' | 'hasta:<nombre de capa>' | 'hasta:<indice>'."""
    if regla == "nada":
        return
    base = next(l for l in nucleo.layers if isinstance(l, keras.Model))   # la red preentrenada
    if regla == "base":
        base.trainable = False
        return
    base.trainable = True
    corte = regla.split(":", 1)[1]
    idx = int(corte) if corte.isdigit() else [l.name for l in base.layers].index(corte)
    for i, capa in enumerate(base.layers):
        # se congelan las capas antes del corte y todas las BatchNorm
        capa.trainable = i >= idx and not isinstance(capa, layers.BatchNormalization)


def optimizador(nombre, lr):
    if nombre == "adam":
        return keras.optimizers.Adam(lr)
    if nombre == "sgd":
        return keras.optimizers.SGD(lr, momentum=0.9)
    raise ValueError(nombre)


# ---------------------------------------------------------------- datos
def cargar_tabla(splits, data_root, por_clase, fraccion, seed):
    df = pd.read_csv(splits)
    raiz = Path(data_root).expanduser()
    df["archivo"] = [str(raiz / r) for r in df["ruta"]]
    clases = sorted(df["clase"].unique())
    df["y"] = df["clase"].map({c: i for i, c in enumerate(clases)})
    partes = {c: df[df["conjunto"] == c] for c in ("train", "val", "test")}
    tr = partes["train"]
    if por_clase:   # curva de aprendizaje: N imagenes por clase, solo en entrenamiento
        tr = tr.sample(frac=1, random_state=seed).groupby("clase").head(por_clase)
    if fraccion < 1:   # pruebas: una fraccion de cada clase
        tr = tr.groupby("clase").sample(frac=fraccion, random_state=seed)
        partes["val"] = partes["val"].groupby("clase").sample(frac=fraccion, random_state=seed)
    partes["train"] = tr
    return partes, clases


def dataset(df, lote, mezclar, seed):
    def leer(ruta, y):
        img = tf.io.decode_image(tf.io.read_file(ruta), channels=3, expand_animations=False)
        img = tf.image.resize(img, (IMG, IMG))   # float32, rango 0-255
        return img, y
    ds = tf.data.Dataset.from_tensor_slices((df["archivo"].values, df["y"].values.astype("int32")))
    if mezclar:
        ds = ds.shuffle(min(len(df), 20000), seed=seed, reshuffle_each_iteration=True)
    return ds.map(leer, num_parallel_calls=tf.data.AUTOTUNE).batch(lote).prefetch(tf.data.AUTOTUNE)


# ---------------------------------------------------------------- evaluacion y graficas
def evaluar(nucleo, df, clases, lote, carpeta):
    probs = nucleo.predict(dataset(df, lote, False, 0), verbose=0)
    y, pred = df["y"].values, probs.argmax(1)
    sana = np.array(["healthy" in c for c in clases])
    enferma_real, enferma_pred = ~sana[y], ~sana[pred]
    top3 = np.mean([yy in np.argsort(-p)[:3] for p, yy in zip(probs, y)])
    m = {
        "imagenes": int(len(y)),
        "accuracy": round(float(accuracy_score(y, pred)), 4),
        "precision_macro": round(float(precision_score(y, pred, average="macro", zero_division=0)), 4),
        "recall_macro": round(float(recall_score(y, pred, average="macro", zero_division=0)), 4),
        "f1_macro": round(float(f1_score(y, pred, average="macro")), 4),
        "top3_accuracy": round(float(top3), 4),
        "recall_enferma": round(float(np.mean(enferma_pred[enferma_real])), 4),
        "confianza_promedio": round(float(probs.max(1).mean()), 4),
    }
    rep = classification_report(y, pred, labels=range(len(clases)), target_names=clases, output_dict=True, zero_division=0)
    (carpeta / "classification_report.json").write_text(json.dumps(rep, indent=1), encoding="utf-8")
    np.save(carpeta / "confusion_matrix.npy", confusion_matrix(y, pred, labels=range(len(clases))))
    # grafica: 9 predicciones incorrectas con su confianza (semana 9)
    errores = np.where(pred != y)[0][:9]
    if len(errores):
        fig, ax = plt.subplots(3, 3, figsize=(10, 10))
        for a, i in zip(ax.flat, errores):
            img = tf.image.resize(tf.io.decode_image(tf.io.read_file(df["archivo"].values[i]), channels=3,
                                                     expand_animations=False), (IMG, IMG)).numpy().astype("uint8")
            a.imshow(img)
            corto = lambda c: c.replace("___", "\n").replace("_", " ")[:40]
            a.set_title(f"Real: {corto(clases[y[i]])}\nPred: {corto(clases[pred[i]])} ({probs[i].max():.0%})",
                        fontsize=7, color="#b23b2a")
        for a in ax.flat:
            a.axis("off")
        fig.suptitle("Predicciones incorrectas (validación)")
        fig.tight_layout()
        fig.savefig(carpeta / "errores.png", dpi=110)
        plt.close(fig)
    return m


def graficar_historial(hist, carpeta, titulo):
    ep = np.arange(1, len(hist) + 1)
    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    ax[0].plot(ep, hist["loss"], label="Entrenamiento")
    ax[0].plot(ep, hist["val_loss"], label="Validación")
    ax[0].set_title("Pérdida")
    ax[1].plot(ep, hist["accuracy"], label="Entrenamiento")
    ax[1].plot(ep, hist["val_accuracy"], label="Validación")
    ax[1].set_title("Accuracy")
    ax[1].set_ylim(0, 1)
    if "etapa" in hist and hist["etapa"].nunique() > 1:   # marca el inicio del fine-tuning
        corte = int(hist.index[hist["etapa"] == 2][0]) + 0.5
        for a in ax:
            a.axvline(corte, color="gray", ls="--", lw=1)
    for a in ax:
        a.set_xlabel("Época")
        a.xaxis.set_major_locator(matplotlib.ticker.MaxNLocator(integer=True))
        a.grid(alpha=0.3)
        a.legend()
        a.spines[["top", "right"]].set_visible(False)
    fig.suptitle(titulo)
    fig.tight_layout()
    fig.savefig(carpeta / "training_curves.png", dpi=120)
    plt.close(fig)


# ---------------------------------------------------------------- principal
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--exp", required=True, choices=sorted(EXPERIMENTOS))
    ap.add_argument("--data-root", required=True, help="carpeta que contiene train/ y valid/")
    ap.add_argument("--splits", default="data/splits.csv")
    ap.add_argument("--reports", default="reports")
    ap.add_argument("--models", default="models/runs")
    ap.add_argument("--pesos", default="imagenet", help="'imagenet' o 'none' (solo para pruebas sin internet)")
    ap.add_argument("--lote", type=int, help="cambia el tamaño de lote (por ejemplo 16 si falta memoria)")
    ap.add_argument("--smoke", action="store_true", help="1 época con el 5 %% de los datos")
    ap.add_argument("--fraccion", type=float, help="fracción de datos (para pruebas)")
    ap.add_argument("--seed", type=int, default=42)
    a = ap.parse_args()

    cfg = EXPERIMENTOS[a.exp]
    keras.utils.set_random_seed(a.seed)
    gpus = tf.config.list_physical_devices("GPU")
    for g in gpus:
        tf.config.experimental.set_memory_growth(g, True)
    if gpus:
        keras.mixed_precision.set_global_policy("mixed_float16")   # Tensor Cores de la RTX
    fraccion = a.fraccion if a.fraccion else (0.05 if a.smoke else 1.0)
    lote = a.lote or cfg["lote"]
    pesos = None if a.pesos == "none" else a.pesos
    sufijo = "_smoke" if (a.smoke or fraccion < 1) else ""
    rep = Path(a.reports) / f"{a.exp}{sufijo}"
    mod = Path(a.models) / f"{a.exp}{sufijo}"
    rep.mkdir(parents=True, exist_ok=True)
    mod.mkdir(parents=True, exist_ok=True)

    partes, clases = cargar_tabla(a.splits, a.data_root, cfg.get("por_clase"), fraccion, a.seed)
    print(f"[{a.exp}] train {len(partes['train'])} | val {len(partes['val'])} | GPU: {[g.name for g in gpus] or 'no'}")
    ds_tr = dataset(partes["train"], lote, True, a.seed)
    ds_va = dataset(partes["val"], lote, False, a.seed)

    if cfg.get("desde"):   # fine-tuning a partir del mejor modelo de otra corrida
        origen = Path(a.models) / f"{cfg['desde']}{sufijo}" / "model.keras"
        if not origen.exists():
            raise SystemExit(f"Primero hay que entrenar {cfg['desde']}: no existe {origen}")
        nucleo = keras.models.load_model(origen)
    else:
        nucleo = construir_nucleo(cfg["arq"], len(clases), pesos)

    entrada = keras.Input((IMG, IMG, 3), name="imagen")
    modelo = keras.Model(entrada, nucleo(capa_aumento(a.seed)(entrada)), name=f"entrenamiento_{a.exp}")

    historiales, t0 = [], time.time()
    for n_etapa, etapa in enumerate(cfg["etapas"], start=1):
        aplicar_congelado(nucleo, etapa["congelar"])
        modelo.compile(optimizer=optimizador(etapa["opt"], etapa["lr"]),
                       loss="sparse_categorical_crossentropy", metrics=["accuracy"])
        entrenables = int(sum(np.prod(v.shape) for v in nucleo.trainable_weights))
        print(f"[{a.exp}] etapa {n_etapa}: {etapa} | parámetros entrenables: {entrenables:,}")
        cb = [
            keras.callbacks.EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True),
            keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.2, patience=2, min_lr=1e-7),
            keras.callbacks.CSVLogger(str(rep / f"history_etapa{n_etapa}.csv")),
        ]
        h = modelo.fit(ds_tr, validation_data=ds_va, epochs=1 if a.smoke else etapa["epocas"],
                       callbacks=cb, verbose=2)
        df_h = pd.DataFrame(h.history)
        df_h["etapa"] = n_etapa
        historiales.append(df_h)
    minutos = (time.time() - t0) / 60

    hist = pd.concat(historiales, ignore_index=True)
    hist.index.name = "epoca"
    hist.to_csv(rep / "history.csv")
    graficar_historial(hist, rep, f"{a.exp}: {cfg['arq']}")
    nucleo.save(mod / "model.keras")
    (mod / "labels.json").write_text(json.dumps(clases, indent=1), encoding="utf-8")

    metricas = evaluar(nucleo, partes["val"], clases, lote, rep)
    metricas.update({
        "experimento": a.exp, "arquitectura": cfg["arq"], "conjunto_evaluado": "val",
        "imagenes_train": int(len(partes["train"])), "epocas": int(len(hist)), "minutos": round(minutos, 1),
        "parametros_totales": int(nucleo.count_params()),
        "accuracy_train_final": round(float(hist["accuracy"].iloc[-1]), 4),
    })
    (rep / "metrics.json").write_text(json.dumps(metricas, indent=2, ensure_ascii=False), encoding="utf-8")
    (rep / "config.json").write_text(json.dumps({**cfg, "lote": lote, "fraccion": fraccion, "seed": a.seed,
                                                 "pesos": a.pesos, "gpu": [g.name for g in gpus],
                                                 "tensorflow": tf.__version__, "keras": keras.__version__,
                                                 "python": platform.python_version()}, indent=2), encoding="utf-8")
    print(json.dumps(metricas, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
