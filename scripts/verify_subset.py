"""Verifica que todas las imagenes de la muestra se puedan abrir y reescribe desde el zip las danadas."""
import argparse, zipfile
from pathlib import Path
from PIL import Image

PREFIX = "New Plant Diseases Dataset(Augmented)/New Plant Diseases Dataset(Augmented)/"

ap = argparse.ArgumentParser()
ap.add_argument("--zip", required=True)
ap.add_argument("--data", default="data/subset")
a = ap.parse_args()
bad = []
for p in Path(a.data).rglob("*"):
    if p.is_file() and p.suffix.lower() in {".jpg", ".jpeg", ".png"}:
        try:
            with Image.open(p) as im:
                im.load()
        except Exception:
            bad.append(p)
print(f"Danadas: {len(bad)}")
if bad:
    with zipfile.ZipFile(Path(a.zip).expanduser()) as z:
        for p in bad:
            rel = p.relative_to(a.data).as_posix()
            name = ("test/test/" + p.name) if rel.startswith("test/") else PREFIX + rel
            p.write_bytes(z.read(name))
            print("Reparada:", rel)
