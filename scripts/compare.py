"""Junta los resultados de todas las corridas en una tabla y genera las graficas comparativas.

Salidas en reports/:
  comparacion.csv y comparacion.md   tabla con las metricas de validacion de cada corrida
  curva_aprendizaje.png              accuracy y F1 segun las imagenes por clase (C1, C2, E2)
  comparacion_modelos.png            accuracy y F1 macro de cada corrida
  evolucion_epocas.png               accuracy de validacion epoca por epoca de todas las corridas

Uso:
  python scripts/compare.py
"""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker
import pandas as pd

ORDEN = ["E1", "C1", "C2", "E2", "E3", "E4", "E5"]
NOMBRES = {
    "E1": "CNN propia (desde cero)",
    "C1": "ResNet50 congelada, 100 img/clase",
    "C2": "ResNet50 congelada, 500 img/clase",
    "E2": "ResNet50 congelada, todas",
    "E3": "ResNet50 fine-tuning, Adam",
    "E4": "ResNet50 fine-tuning, SGD",
    "E5": "MobileNetV2 fine-tuning",
}
VERDE, GRIS = "#2f7d32", "#a9c9aa"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--reports", default="reports")
    ap.add_argument("--sufijo", default="", help="'_smoke' para comparar las pruebas de humo")
    a = ap.parse_args()
    rep = Path(a.reports)

    filas = []
    for e in ORDEN:
        f = rep / f"{e}{a.sufijo}" / "metrics.json"
        if f.exists():
            m = json.loads(f.read_text(encoding="utf-8"))
            filas.append({"ID": e, "Modelo": NOMBRES[e], "Imágenes train": m["imagenes_train"],
                          "Épocas": m["epocas"], "Accuracy": m["accuracy"], "F1 macro": m["f1_macro"],
                          "Top-3": m["top3_accuracy"], "Recall enferma": m["recall_enferma"],
                          "Confianza prom.": m["confianza_promedio"], "Parámetros": m["parametros_totales"],
                          "Minutos": m["minutos"]})
    if not filas:
        raise SystemExit(f"No hay resultados en {rep}/*{a.sufijo}/metrics.json")
    df = pd.DataFrame(filas)
    df.to_csv(rep / f"comparacion{a.sufijo}.csv", index=False)
    md = ["| " + " | ".join(df.columns) + " |", "|" + " --- |" * len(df.columns)]
    for _, r in df.iterrows():
        md.append("| " + " | ".join(f"{v:.1%}" if isinstance(v, float) and k not in ("Minutos",) else str(v)
                                    for k, v in r.items()) + " |")
    (rep / f"comparacion{a.sufijo}.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    print("\n".join(md))

    # comparacion de modelos
    fig, ax = plt.subplots(figsize=(10, 4.5))
    y = range(len(df))[::-1]
    ax.barh([i + 0.2 for i in y], df["Accuracy"] * 100, 0.4, color=VERDE, label="Accuracy")
    ax.barh([i - 0.2 for i in y], df["F1 macro"] * 100, 0.4, color=GRIS, label="F1 macro")
    ax.set_yticks(list(y), [f"{i}: {n}" for i, n in zip(df["ID"], df["Modelo"])])
    ax.set_xlim(0, 100)
    ax.set_xlabel("% en validación")
    ax.axvline(70, color="#b23b2a", ls="--", lw=1)
    ax.text(70.5, len(df) - 0.4, "meta 70 %", color="#b23b2a", fontsize=8)
    for i, v in zip(y, df["Accuracy"] * 100):
        ax.text(v + 0.5, i + 0.2, f"{v:.1f}", va="center", fontsize=8)
    ax.legend(loc="lower right")
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(rep / f"comparacion_modelos{a.sufijo}.png", dpi=130)
    plt.close(fig)

    # curva de aprendizaje
    curva = df[df["ID"].isin(["C1", "C2", "E2"])].sort_values("Imágenes train")
    if len(curva) >= 2:
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.plot(curva["Imágenes train"], curva["Accuracy"] * 100, "o-", color=VERDE, label="Accuracy")
        ax.plot(curva["Imágenes train"], curva["F1 macro"] * 100, "o--", color=GRIS, label="F1 macro")
        for _, r in curva.iterrows():
            ax.annotate(f"{r['Accuracy']:.1%}", (r["Imágenes train"], r["Accuracy"] * 100),
                        textcoords="offset points", xytext=(0, 8), ha="center", fontsize=8)
        ax.set_xscale("log")
        ax.set_xlabel("Imágenes de entrenamiento (escala logarítmica)")
        ax.set_ylabel("% en validación")
        ax.set_title("Curva de aprendizaje: ResNet50 con la base congelada")
        ax.grid(alpha=0.3)
        ax.legend()
        ax.spines[["top", "right"]].set_visible(False)
        fig.tight_layout()
        fig.savefig(rep / f"curva_aprendizaje{a.sufijo}.png", dpi=130)
        plt.close(fig)


    # evolucion epoca por epoca (para ver si mas epocas ayudaban o empeoraban)
    hist = {e: pd.read_csv(rep / f"{e}{a.sufijo}" / "history.csv") for e in df["ID"]
            if (rep / f"{e}{a.sufijo}" / "history.csv").exists()}
    if hist:
        fig, ax = plt.subplots(1, 2, figsize=(13, 4.8))
        for e, h in hist.items():
            ep = range(1, len(h) + 1)
            linea, = ax[0].plot(ep, h["val_accuracy"] * 100, marker="o", ms=3, label=f"{e}: {NOMBRES[e]}")
            ax[1].plot(ep, h["val_loss"], marker="o", ms=3, color=linea.get_color())
            mejor = int(h["val_loss"].idxmin())   # la epoca que se guardo (early stopping)
            ax[0].plot(mejor + 1, h["val_accuracy"].iloc[mejor] * 100, "*", ms=12, color=linea.get_color())
        ax[0].set_title("Accuracy de validación por época (★ = época guardada)")
        ax[0].set_ylabel("%")
        ax[0].set_ylim(0, 100)
        ax[1].set_title("Pérdida de validación por época")
        for x in ax:
            x.set_xlabel("Época")
            x.grid(alpha=0.3)
            x.spines[["top", "right"]].set_visible(False)
            x.xaxis.set_major_locator(matplotlib.ticker.MaxNLocator(integer=True))
        ax[0].legend(fontsize=7, loc="lower right")
        fig.tight_layout()
        fig.savefig(rep / f"evolucion_epocas{a.sufijo}.png", dpi=130)
        plt.close(fig)


if __name__ == "__main__":
    main()
