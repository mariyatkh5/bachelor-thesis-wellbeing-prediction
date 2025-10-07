#!/bin/bash
set -euo pipefail

# ---- Konfig ----
signal_types=( "eda" "ecg" )   # Basis-Signale hier pflegen
scoring="roc_auc"
label="Well-being"
bsl=true    # JSON-Boolean: true/false (nicht True/False)
# -----------------

# EXAKT deine gewünschten Läufe (Analysis -> Test)
runs=(
  "analysis=1,3"
  "analysis=1,8"
  "analysis=3,8"
  "analysis=1"
  "analysis=3"
  "analysis=8"
)

for run in "${runs[@]}"; do
  analysis_csv=$(echo "$run" | sed -n 's/.*analysis=\([^ ]*\).*/\1/p')

  n=${#signal_types[@]}
  max=$(( (1 << n) - 1 ))
  for ((mask=1; mask<=max; mask++)); do
    sig_json="["
    first=true
    for ((i=0; i<n; i++)); do
      if (( (mask >> i) & 1 )); then
        if ! $first; then sig_json+=", "; fi
        sig_json+="\"${signal_types[i]}\""
        first=false
      fi
    done
    sig_json+="]"

    read -r -d '' hyperparameters <<EOF || true
{
  "signal_types": ${sig_json},
  "analysis_datasets": [${analysis_csv}],
  "scoring": "${scoring}",
  "label": "${label}",
  "bsl": ${bsl}
}
EOF

    echo "==> Signals=${sig_json} | Analysis=[${analysis_csv}]"
    echo "finding classifier-----------------------------------------------------------------------------------------------"
    python find_classifier.py "$hyperparameters"
    echo
  done
done
