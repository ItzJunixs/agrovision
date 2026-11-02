"""Genera las graficas del informe a partir de reports/: curvas de entrenamiento y matriz de confusion."""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

rep = Path("reports")
labels = json.loads(Path("models/labels.json").read_text(encoding="utf-8"))
h = json.loads((rep / "history.json").read_text())

fig, ax = plt.subplots(1, 2, figsize=(11, 4))
ep = np.arange(1, len(h["loss"]) + 1)
ax[0].plot(ep, h["loss"], label="train"); ax[0].plot(ep, h["val_loss"], label="valid")
ax[0].set_title("Pérdida (entropía cruzada)"); ax[0].set_xlabel("Época"); ax[0].legend()
ax[1].plot(ep, h["accuracy"], label="train"); ax[1].plot(ep, h["val_accuracy"], label="valid")
ax[1].set_title("Accuracy"); ax[1].set_xlabel("Época"); ax[1].set_ylim(0, 1); ax[1].legend()
for a in ax: a.grid(alpha=.3); a.spines[["top", "right"]].set_visible(False)
fig.tight_layout(); fig.savefig(rep / "training_curves.png", dpi=130)

cm = np.load(rep / "confusion_matrix.npy")
cmn = cm / cm.sum(1, keepdims=True)
short = [l.replace("___", " · ").replace("_", " ")[:34] for l in labels]
fig, ax = plt.subplots(figsize=(14, 12))
im = ax.imshow(cmn, cmap="Greens", vmin=0, vmax=1)
ax.set_xticks(range(len(short)), short, rotation=90, fontsize=7)
ax.set_yticks(range(len(short)), short, fontsize=7)
ax.set_xlabel("Predicción"); ax.set_ylabel("Clase real")
ax.set_title("Matriz de confusión normalizada (validación, 30 imágenes por clase)")
fig.colorbar(im, fraction=0.03)
fig.tight_layout(); fig.savefig(rep / "confusion_matrix.png", dpi=130)
print("ok")
