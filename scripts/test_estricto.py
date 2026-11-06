"""Revision de fuga: evalua los modelos solo con las imagenes de test cuya foto original no
aparece en entrenamiento ni en validacion bajo NINGUNA variacion.

make_split.py reconocia 6 sufijos de variacion (_flipLR, _flipTB, _90deg, _180deg, _270deg,
_new30degFlipLR), pero el dataset trae 8 mas (_newPixel25, _newGRR, _newGGR, _new30degFlipTB,
_new90degFlipLR, _new90degFlipTB, _new200degFlipLR, _new200degFlipTB). Con esos sufijos,
1.421 de las 8.834 imagenes de test tienen una variacion de la misma foto fuera del test.
Este script las excluye y mide de nuevo. splits.csv no se cambia, para que los
entrenamientos sigan siendo reproducibles.

Uso:
  python scripts/test_estricto.py --data-root "~/data/New Plant Diseases Dataset(Augmented)/New Plant Diseases Dataset(Augmented)"
"""
import argparse
import re
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import train  # noqa: E402

import json  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import keras  # noqa: E402
from sklearn.metrics import accuracy_score, f1_score  # noqa: E402

SUFIJOS = re.compile(r"_(new\w+|flipLR|flipTB|180deg|270deg|90deg)$")


def grupo_estricto(ruta, clase):
    stem = Path(ruta).stem
    stem = SUFIJOS.sub("", stem)
    return f"{clase}/{stem.split('___', 1)[1] if '___' in stem else stem}"


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-root", required=True)
    ap.add_argument("--splits", default="data/splits.csv")
    ap.add_argument("--models", default="models/runs")
    ap.add_argument("--reports", default="reports")
    ap.add_argument("--exps", nargs="*", default=["E1", "C1", "C2", "E2", "E3", "E4", "E5"])
    ap.add_argument("--sufijo", default="")
    ap.add_argument("--lote", type=int, default=32)
    a = ap.parse_args()

    df = pd.read_csv(a.splits)
    clases = sorted(df["clase"].unique())
    df["g2"] = [grupo_estricto(r, c) for r, c in zip(df["ruta"], df["clase"])]
    conjuntos = defaultdict(set)
    for g, c in zip(df["g2"], df["conjunto"]):
        conjuntos[g].add(c)
    test = df[df["conjunto"] == "test"].copy()
    test["limpia"] = [conjuntos[g] == {"test"} for g in test["g2"]]
    n, n_limpia = len(test), int(test["limpia"].sum())
    print(f"Test: {n} imágenes | sin ninguna variación fuera del test: {n_limpia} | con variación fuera: {n - n_limpia}")
    test["archivo"] = [str(Path(a.data_root).expanduser() / r) for r in test["ruta"]]
    test["y"] = test["clase"].map({c: i for i, c in enumerate(clases)})
    test = test.reset_index(drop=True)

    filas = []
    for e in a.exps:
        ruta = Path(a.models) / f"{e}{a.sufijo}" / "model.keras"
        if not ruta.exists():
            continue
        nucleo = keras.models.load_model(ruta)
        pred = nucleo.predict(train.dataset(test, a.lote, False, 0), verbose=0).argmax(1)
        y, m = test["y"].values, test["limpia"].values
        filas.append({
            "ID": e,
            "Acc. test completo": round(float(accuracy_score(y, pred)), 4),
            "Acc. test estricto": round(float(accuracy_score(y[m], pred[m])), 4),
            "F1 test estricto": round(float(f1_score(y[m], pred[m], average="macro")), 4),
            "Acc. imágenes con variación fuera": round(float(accuracy_score(y[~m], pred[~m])), 4) if (~m).any() else None,
        })
        print(filas[-1])
        keras.backend.clear_session()
    tabla = pd.DataFrame(filas)
    sal = Path(a.reports)
    tabla.to_csv(sal / f"test_estricto{a.sufijo}.csv", index=False)
    md = [f"Test estricto: {n_limpia} de {n} imágenes de test, sin ninguna variación de la misma foto en entrenamiento o validación.", "",
          "| " + " | ".join(tabla.columns) + " |", "|" + " --- |" * len(tabla.columns)]
    for _, r in tabla.iterrows():
        md.append("| " + " | ".join(str(r[c]) if c == "ID" else f"{r[c]:.1%}" for c in tabla.columns) + " |")
    (sal / f"test_estricto{a.sufijo}.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    print("\n".join(md))


if __name__ == "__main__":
    main()
