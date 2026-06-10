#!/usr/bin/env bash
set -euo pipefail

usage() {
    cat <<'EOF'
Usage:
  scripts/hf-jobs/launch-jianyang-phase1.sh [--confirm]

Dry-runs by default. Pass --confirm to submit the spend-bearing Hugging Face
Job.

Environment overrides:
  HF_IMAGE                 Docker image, default ubuntu:24.04
  HF_FLAVOR                HF Jobs flavor, default h200x8
  HF_TIMEOUT               Job timeout, default 24h
  HF_NAMESPACE             Job namespace, default meshllm
  HF_BUCKET                Writable bucket mounted at /bucket, default meshllm/layer-split-output

  SOURCE_REPO              default zai-org/GLM-5.1
  SOURCE_REVISION          default 26e1bd6e011feb778d25ae34b09b07074139d92d
  QUANT_TYPE               default Q3_K_M
  INTERMEDIATE_OUTTYPE     default bf16
  CONVERT_SPLIT_MAX_SIZE   default 48G
  MESH_LLM_REF             default feat/jianyang
  TARGET_GGUF_REPO         default meshllm/GLM-5.1-Q3_K_M-jianyang-GGUF
  TARGET_PACKAGE_REPO      default meshllm/GLM-5.1-Q3_K_M-jianyang-layers
EOF
}

CONFIRM=false
while [ "$#" -gt 0 ]; do
    case "$1" in
        --confirm)
            CONFIRM=true
            shift
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        *)
            echo "ERROR: unknown argument: $1" >&2
            usage >&2
            exit 2
            ;;
    esac
done

require_command() {
    local command_name="$1"
    if ! command -v "$command_name" >/dev/null 2>&1; then
        echo "ERROR: required command not found: $command_name" >&2
        exit 1
    fi
}

cost_per_hour() {
    case "$1" in
        h200x8) echo "40.00" ;;
        h200x4) echo "20.00" ;;
        l40sx8) echo "23.50" ;;
        a100x8) echo "20.00" ;;
        rtx-pro-6000x8) echo "22.00" ;;
        *) echo "" ;;
    esac
}

timeout_hours() {
    python3 - "$1" <<'PY'
import re
import sys

value = sys.argv[1].strip()
match = re.fullmatch(r"([0-9]+(?:\.[0-9]+)?)([smhd]?)", value)
if not match:
    print("")
    raise SystemExit(0)

amount = float(match.group(1))
unit = match.group(2) or "s"
factor = {"s": 1 / 3600, "m": 1 / 60, "h": 1, "d": 24}[unit]
print(amount * factor)
PY
}

print_command() {
    local arg
    printf 'Command:'
    for arg in "$@"; do
        printf ' %q' "$arg"
    done
    printf '\n'
}

require_command hf
require_command python3

HF_IMAGE="${HF_IMAGE:-ubuntu:24.04}"
HF_FLAVOR="${HF_FLAVOR:-h200x8}"
HF_TIMEOUT="${HF_TIMEOUT:-24h}"
HF_NAMESPACE="${HF_NAMESPACE:-meshllm}"
HF_BUCKET="${HF_BUCKET:-meshllm/layer-split-output}"

SOURCE_REPO="${SOURCE_REPO:-zai-org/GLM-5.1}"
SOURCE_REVISION="${SOURCE_REVISION:-26e1bd6e011feb778d25ae34b09b07074139d92d}"
QUANT_TYPE="${QUANT_TYPE:-Q3_K_M}"
INTERMEDIATE_OUTTYPE="${INTERMEDIATE_OUTTYPE:-bf16}"
CONVERT_SPLIT_MAX_SIZE="${CONVERT_SPLIT_MAX_SIZE:-48G}"
MESH_LLM_REF="${MESH_LLM_REF:-feat/jianyang}"
TARGET_GGUF_REPO="${TARGET_GGUF_REPO:-meshllm/GLM-5.1-Q3_K_M-jianyang-GGUF}"
TARGET_PACKAGE_REPO="${TARGET_PACKAGE_REPO:-meshllm/GLM-5.1-Q3_K_M-jianyang-layers}"
MODEL_ID="${MODEL_ID:-${TARGET_PACKAGE_REPO}}"
GGUF_SUBDIR="${GGUF_SUBDIR:-${QUANT_TYPE}}"
GGUF_BASENAME="${GGUF_BASENAME:-GLM-5.1-${QUANT_TYPE}}"
MODEL_NAME="${MODEL_NAME:-GLM-5.1}"
KEEP_OUTPUT_TENSOR="${KEEP_OUTPUT_TENSOR:-true}"
TOKEN_EMBEDDING_TYPE="${TOKEN_EMBEDDING_TYPE:-q8_0}"
TENSOR_TYPE_OVERRIDES="${TENSOR_TYPE_OVERRIDES:-mtp=q8_0 nextn=q8_0}"
WORK_ROOT="${WORK_ROOT:-/bucket/jianyang-quant-package}"
CATALOG_CREATE_PR="${CATALOG_CREATE_PR:-false}"

BOOTSTRAP=$(cat <<'BASH'
set -euo pipefail
apt-get update -qq
DEBIAN_FRONTEND=noninteractive apt-get install -y -qq ca-certificates git > /dev/null
apt-get clean
rm -rf /var/lib/apt/lists/*
git clone --filter=blob:none https://github.com/Mesh-LLM/mesh-llm.git /tmp/mesh-llm
cd /tmp/mesh-llm
git fetch --depth 1 origin "$MESH_LLM_REF"
git checkout --detach FETCH_HEAD
bash scripts/hf-jobs/checkpoint-quant-package-job.sh
BASH
)

HF_ARGS=(
    jobs run
    --detach
    --namespace "$HF_NAMESPACE"
    --flavor "$HF_FLAVOR"
    --timeout "$HF_TIMEOUT"
    --volume "hf://buckets/${HF_BUCKET}:/bucket"
    --secrets HF_TOKEN
    --label jianyang
    --label phase=1
    --label model=glm-5.1
    --env "SOURCE_REPO=${SOURCE_REPO}"
    --env "SOURCE_REVISION=${SOURCE_REVISION}"
    --env "QUANT_TYPE=${QUANT_TYPE}"
    --env "INTERMEDIATE_OUTTYPE=${INTERMEDIATE_OUTTYPE}"
    --env "CONVERT_SPLIT_MAX_SIZE=${CONVERT_SPLIT_MAX_SIZE}"
    --env "MESH_LLM_REF=${MESH_LLM_REF}"
    --env "TARGET_GGUF_REPO=${TARGET_GGUF_REPO}"
    --env "TARGET_PACKAGE_REPO=${TARGET_PACKAGE_REPO}"
    --env "MODEL_ID=${MODEL_ID}"
    --env "GGUF_SUBDIR=${GGUF_SUBDIR}"
    --env "GGUF_BASENAME=${GGUF_BASENAME}"
    --env "MODEL_NAME=${MODEL_NAME}"
    --env "KEEP_OUTPUT_TENSOR=${KEEP_OUTPUT_TENSOR}"
    --env "TOKEN_EMBEDDING_TYPE=${TOKEN_EMBEDDING_TYPE}"
    --env "TENSOR_TYPE_OVERRIDES=${TENSOR_TYPE_OVERRIDES}"
    --env "WORK_ROOT=${WORK_ROOT}"
    --env "CATALOG_CREATE_PR=${CATALOG_CREATE_PR}"
    "$HF_IMAGE"
    /bin/bash -lc "$BOOTSTRAP"
)

echo "Jianyang Phase 1 HF Job"
echo "  source:  ${SOURCE_REPO}@${SOURCE_REVISION}"
echo "  quant:   ${QUANT_TYPE}"
echo "  branch:  ${MESH_LLM_REF}"
echo "  GGUF:    ${TARGET_GGUF_REPO}"
echo "  package: ${TARGET_PACKAGE_REPO}"
echo "  flavor:  ${HF_FLAVOR}"
echo "  timeout: ${HF_TIMEOUT}"
echo "  bucket:  ${HF_BUCKET}"

rate="$(cost_per_hour "$HF_FLAVOR")"
hours="$(timeout_hours "$HF_TIMEOUT")"
if [ -n "$rate" ] && [ -n "$hours" ]; then
    python3 - "$rate" "$hours" <<'PY'
import sys
rate = float(sys.argv[1])
hours = float(sys.argv[2])
print(f"  max cost at timeout: ${rate * hours:,.2f}")
PY
else
    echo "  max cost at timeout: unknown for this flavor/timeout"
fi

print_command hf "${HF_ARGS[@]}"

if [ "$CONFIRM" != "true" ]; then
    echo
    echo "Dry run only. Re-run with --confirm to submit this spend-bearing job."
    exit 0
fi

echo
echo "Submitting spend-bearing Hugging Face Job..."
exec hf "${HF_ARGS[@]}"
