"""Extrae una muestra balanceada del zip de Kaggle sin descomprimirlo completo.

Uso:
    python scripts/extract_subset.py --zip ~/Downloads/archive.zip --out data/subset --train 100 --valid 30
"""
import argparse, random, zipfile
from collections import defaultdict
from pathlib import Path

PREFIX = "New Plant Diseases Dataset(Augmented)/New Plant Diseases Dataset(Augmented)/"  # el zip trae una copia duplicada; usamos solo esta

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--zip", required=True)
    ap.add_argument("--out", default="data/subset")
    ap.add_argument("--train", type=int, default=100, help="imagenes por clase para train")
    ap.add_argument("--valid", type=int, default=30, help="imagenes por clase para valid")
    ap.add_argument("--seed", type=int, default=42)
    a = ap.parse_args()
    random.seed(a.seed)
    out = Path(a.out)
    with zipfile.ZipFile(Path(a.zip).expanduser()) as z:
        groups = defaultdict(list)
        for n in z.namelist():
            if n.startswith(PREFIX) and not n.endswith("/"):
                split, cls, _ = n[len(PREFIX):].split("/", 2)
                groups[(split, cls)].append(n)
        total = 0
        for (split, cls), files in sorted(groups.items()):
            k = a.train if split == "train" else a.valid
            for n in random.sample(sorted(files), min(k, len(files))):
                dest = out / split / cls / Path(n).name
                if dest.exists():
                    continue  # permite reanudar si se interrumpe
                dest.parent.mkdir(parents=True, exist_ok=True)
                dest.write_bytes(z.read(n))
                total += 1
        # carpeta test original (33 imagenes, sin subcarpetas por clase)
        for n in z.namelist():
            if n.startswith("test/test/") and not n.endswith("/"):
                dest = out / "test" / Path(n).name
                dest.parent.mkdir(parents=True, exist_ok=True)
                dest.write_bytes(z.read(n)); total += 1
    print(f"Extraidas {total} imagenes en {out}")

if __name__ == "__main__":
    main()
