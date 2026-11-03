#!/usr/bin/env bash
# Lanza los experimentos del roadmap uno tras otro (pensado para WSL con GPU).
#
# Uso (desde la carpeta del repo, con el entorno activado: source ~/venv/bin/activate):
#   bash scripts/run_all.sh smoke            # prueba de humo de todos: 1 época con el 5 % de los datos
#   bash scripts/run_all.sh                  # todos: E1 C1 C2 E2 E3 E4 E5
#   bash scripts/run_all.sh E2 E3            # solo los indicados
#
# La ruta del dataset se puede cambiar con la variable DATA_ROOT.
set -euo pipefail

DATA_ROOT="${DATA_ROOT:-$HOME/data/New Plant Diseases Dataset(Augmented)/New Plant Diseases Dataset(Augmented)}"
EXTRA=()
if [[ "${1:-}" == "smoke" ]]; then
  EXTRA=(--smoke)
  shift
fi
EXPS=("$@")
if [[ ${#EXPS[@]} -eq 0 ]]; then
  EXPS=(E1 C1 C2 E2 E3 E4 E5)   # E3 y E4 parten de E2, por eso van despues
fi

mkdir -p logs
for e in "${EXPS[@]}"; do
  echo "================ $e  ($(date '+%H:%M')) ================"
  python scripts/train.py --exp "$e" --data-root "$DATA_ROOT" "${EXTRA[@]}" 2>&1 | tee "logs/$e${EXTRA:+_smoke}.log"
done
python scripts/compare.py ${EXTRA:+--sufijo _smoke}
echo "Listo ($(date '+%H:%M')). Resultados en reports/ y logs/."
