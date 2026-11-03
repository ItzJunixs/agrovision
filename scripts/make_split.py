"""Divide el dataset en entrenamiento, validacion y prueba (80/10/10) sin fuga de datos.

El dataset de Kaggle trae variaciones de cada foto original (espejos y rotaciones) y muchas
de las imagenes de `valid` son variaciones de fotos que tambien estan en `train`. Este script:

1. Junta `train` y `valid` (87.867 imagenes).
2. Agrupa cada imagen con su foto original: el nombre del archivo sin el identificador inicial
   ni el sufijo de variacion (por ejemplo `FREC_Scab 3335`).
3. Divide por grupos, clase por clase, para que todas las variaciones de una foto queden en el
   mismo conjunto y las 38 clases mantengan la proporcion 80/10/10.
4. Verifica que ningun grupo quede en dos conjuntos y guarda data/splits.csv.

Funciona leyendo el zip directamente (sin extraerlo) o una carpeta ya extraida.

Uso:
  python scripts/make_split.py --zip ~/archive.zip
  python scripts/make_split.py --data-dir "~/data/New Plant Diseases Dataset(Augmented)/New Plant Diseases Dataset(Augmented)"
"""
import argparse
import csv
import json
import random
import re
import zipfile
from collections import Counter, defaultdict
from pathlib import Path

PREFIX = "New Plant Diseases Dataset(Augmented)/New Plant Diseases Dataset(Augmented)/"
SUFIJOS = re.compile(r"_(new30degFlipLR|flipLR|flipTB|180deg|270deg|90deg)$")
EXT = {".jpg", ".jpeg", ".png"}


def grupo_de(nombre: str) -> tuple[str, bool]:
    """Devuelve (foto original, es_variacion) a partir del nombre del archivo."""
    stem = Path(nombre).stem
    es_variacion = bool(SUFIJOS.search(stem))
    stem = SUFIJOS.sub("", stem)
    return (stem.split("___", 1)[1] if "___" in stem else stem), es_variacion


def listar(args) -> list[tuple[str, str, str]]:
    """Lista (origen, clase, archivo) de train y valid."""
    filas = []
    if args.zip:
        with zipfile.ZipFile(Path(args.zip).expanduser()) as z:
            for n in z.namelist():
                if n.startswith(PREFIX) and not n.endswith("/"):
                    origen, clase, archivo = n[len(PREFIX):].split("/", 2)
                    if origen in ("train", "valid") and Path(archivo).suffix.lower() in EXT:
                        filas.append((origen, clase, archivo))
    else:
        raiz = Path(args.data_dir).expanduser()
        for origen in ("train", "valid"):
            for p in sorted((raiz / origen).glob("*/*")):
                if p.suffix.lower() in EXT:
                    filas.append((origen, p.parent.name, p.name))
    return filas


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    fuente = ap.add_mutually_exclusive_group(required=True)
    fuente.add_argument("--zip", help="ruta a archive.zip")
    fuente.add_argument("--data-dir", help="carpeta con las subcarpetas train/ y valid/")
    ap.add_argument("--out", default="data/splits.csv")
    ap.add_argument("--val", type=float, default=0.10)
    ap.add_argument("--test", type=float, default=0.10)
    ap.add_argument("--seed", type=int, default=42)
    a = ap.parse_args()

    filas = listar(a)
    if not filas:
        raise SystemExit("No se encontraron imagenes: revisa la ruta.")

    # clase -> grupo -> lista de (origen, archivo, es_variacion)
    por_clase = defaultdict(lambda: defaultdict(list))
    for origen, clase, archivo in filas:
        g, var = grupo_de(archivo)
        por_clase[clase][g].append((origen, archivo, var))

    rng = random.Random(a.seed)
    salida = []
    for clase in sorted(por_clase):
        grupos = sorted(por_clase[clase])
        rng.shuffle(grupos)
        total = sum(len(por_clase[clase][g]) for g in grupos)
        meta_test, meta_val = a.test * total, a.val * total
        n_test = n_val = 0
        for g in grupos:
            imgs = por_clase[clase][g]
            # se llena primero test, luego validacion; el resto va a entrenamiento
            if n_test < meta_test:
                conjunto, n_test = "test", n_test + len(imgs)
            elif n_val < meta_val:
                conjunto, n_val = "val", n_val + len(imgs)
            else:
                conjunto = "train"
            for origen, archivo, var in imgs:
                salida.append({"ruta": f"{origen}/{clase}/{archivo}", "clase": clase,
                               "grupo": f"{clase}/{g}", "conjunto": conjunto,
                               "origen_kaggle": origen, "es_variacion": int(var)})

    # verificaciones
    conj_por_grupo = defaultdict(set)
    for f in salida:
        conj_por_grupo[f["grupo"]].add(f["conjunto"])
    compartidos = sum(1 for s in conj_por_grupo.values() if len(s) > 1)
    clases = sorted(por_clase)
    for c in ("train", "val", "test"):
        presentes = {f["clase"] for f in salida if f["conjunto"] == c}
        assert presentes == set(clases), f"Faltan clases en {c}: {set(clases) - presentes}"
    assert compartidos == 0, f"{compartidos} grupos aparecen en mas de un conjunto"

    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(salida[0]))
        w.writeheader()
        w.writerows(salida)

    cuenta = Counter(f["conjunto"] for f in salida)
    n = len(salida)
    resumen = {
        "imagenes": n, "clases": len(clases), "grupos": len(conj_por_grupo),
        "grupos_compartidos_entre_conjuntos": compartidos, "semilla": a.seed,
        "por_conjunto": {c: {"imagenes": cuenta[c], "porcentaje": round(100 * cuenta[c] / n, 1)}
                         for c in ("train", "val", "test")},
        "por_clase": {cl: dict(Counter(f["conjunto"] for f in salida if f["clase"] == cl)) for cl in clases},
    }
    out.with_name("splits_resumen.json").write_text(json.dumps(resumen, indent=1, ensure_ascii=False), encoding="utf-8")
    print(f"{n} imagenes, {len(clases)} clases, {len(conj_por_grupo)} fotos originales")
    for c in ("train", "val", "test"):
        print(f"  {c:5s}: {cuenta[c]:6d} ({100 * cuenta[c] / n:.1f} %)")
    print(f"Grupos compartidos entre conjuntos: {compartidos}")
    print(f"Guardado en {out}")


if __name__ == "__main__":
    main()
