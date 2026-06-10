#!/usr/bin/env bash
set -euo pipefail

# Hugging Face Jobs entrypoint for producing a custom GGUF quant from a
# checkpoint repository, publishing that GGUF, and publishing a Skippy layer
# package derived from it.
#
# Required secret:
#   HF_TOKEN
#
# Important environment variables:
#   SOURCE_REPO              HF checkpoint repo, e.g. zai-org/GLM-5.1
#   SOURCE_REVISION          source commit/revision
#   TARGET_GGUF_REPO         output GGUF repo, e.g. meshllm/GLM-5.1-Q3_K_M-jianyang-GGUF
#   TARGET_PACKAGE_REPO      output package repo, e.g. meshllm/GLM-5.1-Q3_K_M-jianyang-layers
#   MODEL_ID                 model id exposed by mesh-llm
#   QUANT_TYPE               llama.cpp quant type, e.g. Q3_K_M
#   MESH_LLM_REF             mesh-llm git ref used by downstream package job
#
# Optional:
#   JOB_MODE                 full or conversion-plan, default full
#   INTERMEDIATE_OUTTYPE     converter output type, default bf16
#   CONVERT_SPLIT_MAX_SIZE   converter split size, default 48G
#   GGUF_SUBDIR              subdir in output GGUF repo, default QUANT_TYPE
#   GGUF_BASENAME            output basename before shard suffixes
#   MODEL_NAME               model name written by convert_hf_to_gguf.py
#   KEEP_OUTPUT_TENSOR       true keeps output.weight unquantized
#   TOKEN_EMBEDDING_TYPE     default q8_0
#   TENSOR_TYPE_OVERRIDES    newline or space separated llama-quantize tensor-type overrides
#   WORK_ROOT                writable workspace, default /bucket/jianyang-quant-package
#   PACKAGE_SCRIPT           package script path, default crates/model-package split script
#   ESTIMATE_COST_PER_HOUR   optional cost rate used in conversion-plan estimate

require_env() {
    local name="$1"
    if [ -z "${!name:-}" ]; then
        echo "ERROR: $name is required" >&2
        exit 1
    fi
}

require_env HF_TOKEN

JOB_MODE="${JOB_MODE:-full}"
case "$JOB_MODE" in
    full|conversion-plan) ;;
    *)
        echo "ERROR: unsupported JOB_MODE: $JOB_MODE" >&2
        echo "Expected one of: full, conversion-plan" >&2
        exit 1
        ;;
esac

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
JOB_ID_SUFFIX="$(date -u +%Y%m%dT%H%M%SZ)-$$"
WORK_DIR="${WORK_DIR:-${WORK_ROOT}/${JOB_ID_SUFFIX}}"
BUILD_DIR="${BUILD_DIR:-${WORK_DIR}/mesh-llm}"
GGUF_DIR="${GGUF_DIR:-${WORK_DIR}/gguf}"
BF16_DIR="${BF16_DIR:-${GGUF_DIR}/intermediate-${INTERMEDIATE_OUTTYPE}}"
QUANT_DIR="${QUANT_DIR:-${GGUF_DIR}/${GGUF_SUBDIR}}"
HF_HOME="${HF_HOME:-${WORK_DIR}/hf-home}"
HF_HUB_CACHE="${HF_HUB_CACHE:-${HF_HOME}/hub}"
HF_XET_CACHE="${HF_XET_CACHE:-${HF_HOME}/xet}"
TMPDIR="${TMPDIR:-${WORK_DIR}/tmp}"
PACKAGE_SCRIPT="${PACKAGE_SCRIPT:-${BUILD_DIR}/crates/model-package/src/scripts/split-model-job.sh}"

export HF_HOME HF_HUB_CACHE HF_XET_CACHE TMPDIR TEMP="$TMPDIR" TMP="$TMPDIR"

mkdir -p "$WORK_DIR" "$BF16_DIR" "$QUANT_DIR" "$HF_HUB_CACHE" "$HF_XET_CACHE" "$TMPDIR"

log_step() {
    echo
    echo "=== $* ==="
}

format_bytes() {
    python3 - "$1" <<'PY'
import sys
value = float(int(sys.argv[1]))
for unit in ["B", "KiB", "MiB", "GiB", "TiB", "PiB"]:
    if value < 1024 or unit == "PiB":
        print(f"{value:.1f} {unit}" if unit != "B" else f"{int(value)} {unit}")
        break
    value /= 1024
PY
}

print_conversion_plan_estimate() {
    log_step "Estimate full quant/package lower bound"
    python3 - <<'PY'
from huggingface_hub import HfApi
import os

source_repo = os.environ["SOURCE_REPO"]
source_revision = os.environ["SOURCE_REVISION"]
quant_type = os.environ["QUANT_TYPE"]
cost_per_hour = os.environ.get("ESTIMATE_COST_PER_HOUR", "")

quant_ratios = {
    "Q2_K": 0.19,
    "Q3_K_S": 0.21,
    "Q3_K_M": 0.23,
    "Q3_K_L": 0.25,
    "Q4_K_S": 0.28,
    "Q4_K_M": 0.31,
    "Q5_K_M": 0.38,
    "Q8_0": 0.53,
}

def fmt_bytes(value: float) -> str:
    units = ["B", "KiB", "MiB", "GiB", "TiB"]
    unit = units[0]
    for unit in units:
        if abs(value) < 1024 or unit == units[-1]:
            break
        value /= 1024
    if unit == "B":
        return f"{int(value)} {unit}"
    return f"{value:.1f} {unit}"

api = HfApi(token=os.environ.get("HF_TOKEN"))
info = api.model_info(
    repo_id=source_repo,
    revision=source_revision,
    files_metadata=True,
)

source_bytes = 0
for sibling in info.siblings or []:
    name = getattr(sibling, "rfilename", "")
    size = getattr(sibling, "size", None)
    if name.endswith(".safetensors") and size:
        source_bytes += int(size)

if source_bytes <= 0:
    print("Could not estimate source safetensor bytes from Hub metadata.")
    raise SystemExit(0)

ratio = quant_ratios.get(quant_type, 0.30)
bf16_bytes = source_bytes
quant_bytes = source_bytes * ratio

# Lower-bound bytes moved by the full path:
# - conversion reads source safetensors and writes BF16 GGUF
# - quantization reads BF16 GGUF and writes quant GGUF
# - GGUF upload reads/uploads quant GGUF
# - package creation reads quant GGUF and uploads layer artifacts of about the
#   same order as the quant GGUF
lower_bound_io = source_bytes + bf16_bytes + bf16_bytes + quant_bytes * 4

print(f"Source safetensor bytes: {fmt_bytes(source_bytes)}")
print(f"Estimated BF16 intermediate bytes: {fmt_bytes(bf16_bytes)}")
print(f"Estimated {quant_type} GGUF bytes at ratio {ratio:.2f}: {fmt_bytes(quant_bytes)}")
print(f"Estimated full-job I/O lower bound: {fmt_bytes(lower_bound_io)}")
print("This excludes CPU quantization overhead, retries, Hub throttling, and package validation.")

try:
    rate = float(cost_per_hour) if cost_per_hour else None
except ValueError:
    rate = None

for mbps in [25, 50, 100, 200]:
    hours = lower_bound_io / (mbps * 1000 * 1000) / 3600
    if rate is None:
        print(f"At sustained {mbps} MB/s I/O: lower-bound runtime {hours:.1f}h")
    else:
        print(f"At sustained {mbps} MB/s I/O: lower-bound runtime {hours:.1f}h, cost ${hours * rate:.2f}")
PY
}

storage_snapshot() {
    echo "Workspace: $WORK_DIR"
    df -h / /bucket "$WORK_DIR" "$TMPDIR" 2>/dev/null || true
    du -sh "$WORK_DIR" 2>/dev/null || true
}

on_error() {
    local rc=$?
    echo "ERROR: command failed with status $rc: ${BASH_COMMAND}" >&2
    storage_snapshot >&2 || true
    exit "$rc"
}
trap on_error ERR

log_step "Phase 1 checkpoint quant/package job"
echo "Source checkpoint: ${SOURCE_REPO}@${SOURCE_REVISION}"
echo "Target GGUF repo:   ${TARGET_GGUF_REPO}"
echo "Target package:     ${TARGET_PACKAGE_REPO}"
echo "Quant type:         ${QUANT_TYPE}"
echo "Mesh LLM ref:       ${MESH_LLM_REF}"
echo "Job mode:           ${JOB_MODE}"
storage_snapshot

log_step "Install system dependencies"
apt-get update -qq
DEBIAN_FRONTEND=noninteractive apt-get install -y -qq \
    build-essential ca-certificates cmake curl git ninja-build pkg-config \
    python3-pip python3-venv libssl-dev > /dev/null
apt-get clean
rm -rf /var/lib/apt/lists/*

log_step "Clone mesh-llm"
if [ ! -d "$BUILD_DIR/.git" ]; then
    git clone --filter=blob:none https://github.com/Mesh-LLM/mesh-llm.git "$BUILD_DIR"
fi
cd "$BUILD_DIR"
git fetch --depth 1 origin "$MESH_LLM_REF"
git checkout --detach FETCH_HEAD

log_step "Prepare patched llama.cpp"
scripts/prepare-llama.sh pinned

if [ "$JOB_MODE" = "full" ]; then
    log_step "Build llama.cpp libraries and quantizer"
    scripts/build-llama.sh -DLLAMA_BUILD_TOOLS=ON
    LLAMA_BUILD_DIR="$(scripts/build-llama.sh --print-build-dir)"
    cmake --build "$LLAMA_BUILD_DIR" --config Release --parallel "$(nproc)" --target llama-quantize
    LLAMA_QUANTIZE="${LLAMA_BUILD_DIR}/bin/llama-quantize"
    test -x "$LLAMA_QUANTIZE"

    log_step "Build skippy-model-package"
    SKIPPY_LLAMA_BUILD_DIR="$LLAMA_BUILD_DIR" cargo build --release -p skippy-model-package
    SLICER="${CARGO_TARGET_DIR:-${BUILD_DIR}/target}/release/skippy-model-package"
    test -x "$SLICER"
fi

log_step "Prepare Python conversion environment"
python3 -m venv "${WORK_DIR}/venv"
# shellcheck disable=SC1091
source "${WORK_DIR}/venv/bin/activate"
pip install -q --upgrade pip
pip install -q -r .deps/llama.cpp/requirements/requirements-convert_hf_to_gguf.txt
pip install -q huggingface_hub hf_xet

log_step "Pin llama.cpp remote conversion to source revision"
python3 - "$SOURCE_REVISION" <<'PY'
from pathlib import Path
import sys

revision = sys.argv[1]

converter = Path(".deps/llama.cpp/convert_hf_to_gguf.py")
converter_text = converter.read_text()
old_snapshot = """local_dir = snapshot_download(
            repo_id=hf_repo_id,
            allow_patterns=allowed_patterns)"""
new_snapshot = f"""local_dir = snapshot_download(
            repo_id=hf_repo_id,
            revision={revision!r},
            allow_patterns=allowed_patterns)"""
if old_snapshot not in converter_text:
    raise SystemExit("could not find snapshot_download block to patch")
converter.write_text(converter_text.replace(old_snapshot, new_snapshot))

utility = Path(".deps/llama.cpp/gguf-py/gguf/utility.py")
utility_text = utility.read_text()
if "/resolve/main/" not in utility_text:
    raise SystemExit("could not find remote safetensor resolve path to patch")
utility.write_text(utility_text.replace("/resolve/main/", f"/resolve/{revision}/"))
PY

log_step "Convert checkpoint to split ${INTERMEDIATE_OUTTYPE} GGUF"
BF16_OUT="${BF16_DIR}/${GGUF_BASENAME}-${INTERMEDIATE_OUTTYPE}.gguf"
CONVERT_ARGS=(
    --remote "$SOURCE_REPO" \
    --outfile "$BF16_OUT" \
    --outtype "$INTERMEDIATE_OUTTYPE" \
    --split-max-size "$CONVERT_SPLIT_MAX_SIZE" \
    --model-name "$MODEL_NAME"
)

if [ "$JOB_MODE" = "conversion-plan" ]; then
    python .deps/llama.cpp/convert_hf_to_gguf.py "${CONVERT_ARGS[@]}" --dry-run
    print_conversion_plan_estimate
    log_step "Conversion-plan job complete"
    exit 0
fi

python .deps/llama.cpp/convert_hf_to_gguf.py "${CONVERT_ARGS[@]}"

BF16_FIRST="$(find "$BF16_DIR" -maxdepth 1 -type f -name '*.gguf' | sort | head -1)"
if [ -z "$BF16_FIRST" ]; then
    echo "ERROR: converter emitted no GGUF shards in $BF16_DIR" >&2
    exit 1
fi
echo "First intermediate shard: $BF16_FIRST"
du -sh "$BF16_DIR"

log_step "Quantize split GGUF to ${QUANT_TYPE}"
QUANT_OUT="${QUANT_DIR}/${GGUF_BASENAME}-00001-of-00001.gguf"
QUANT_ARGS=()
if [ "$KEEP_OUTPUT_TENSOR" = "true" ]; then
    QUANT_ARGS+=(--leave-output-tensor)
fi
if [ -n "$TOKEN_EMBEDDING_TYPE" ]; then
    QUANT_ARGS+=(--token-embedding-type "$TOKEN_EMBEDDING_TYPE")
fi
if [ -n "$TENSOR_TYPE_OVERRIDES" ]; then
    while IFS= read -r override; do
        [ -n "$override" ] || continue
        QUANT_ARGS+=(--tensor-type "$override")
    done < <(printf '%s\n' "$TENSOR_TYPE_OVERRIDES" | tr '[:space:]' '\n')
fi
"$LLAMA_QUANTIZE" \
    --keep-split \
    "${QUANT_ARGS[@]}" \
    "$BF16_FIRST" \
    "$QUANT_OUT" \
    "$QUANT_TYPE" \
    "$(nproc)"

QUANT_FIRST="$(find "$QUANT_DIR" -maxdepth 1 -type f -name '*.gguf' | sort | head -1)"
if [ -z "$QUANT_FIRST" ]; then
    echo "ERROR: quantizer emitted no GGUF shards in $QUANT_DIR" >&2
    exit 1
fi
echo "First quant shard: $QUANT_FIRST"
du -sh "$QUANT_DIR"

log_step "Remove intermediate GGUF shards"
rm -rf "$BF16_DIR"
storage_snapshot

log_step "Publish custom GGUF shards"
export TARGET_GGUF_REPO QUANT_DIR GGUF_SUBDIR SOURCE_REPO SOURCE_REVISION QUANT_TYPE
export INTERMEDIATE_OUTTYPE CONVERT_SPLIT_MAX_SIZE TENSOR_TYPE_OVERRIDES
export KEEP_OUTPUT_TENSOR TOKEN_EMBEDDING_TYPE MODEL_NAME
GGUF_REVISION_FILE="${WORK_DIR}/target-gguf-revision.txt"
export GGUF_REVISION_FILE
python3 - <<'PY'
from pathlib import Path
from huggingface_hub import HfApi
import hashlib
import json
import os

api = HfApi(token=os.environ["HF_TOKEN"])
repo = os.environ["TARGET_GGUF_REPO"]
quant_dir = Path(os.environ["QUANT_DIR"])
subdir = os.environ["GGUF_SUBDIR"]
source_repo = os.environ["SOURCE_REPO"]
source_revision = os.environ["SOURCE_REVISION"]
quant_type = os.environ["QUANT_TYPE"]

api.create_repo(repo, repo_type="model", exist_ok=True)

files = []
for path in sorted(quant_dir.glob("*.gguf")):
    rel = f"{subdir}/{path.name}"
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    size = path.stat().st_size
    api.upload_file(
        repo_id=repo,
        repo_type="model",
        path_or_fileobj=str(path),
        path_in_repo=rel,
        commit_message=f"Add {quant_type} GGUF shard {path.name}",
    )
    files.append({"path": rel, "bytes": size, "sha256": digest.hexdigest()})
    print(f"uploaded {rel} ({size} bytes)")

manifest = {
    "schema_version": 1,
    "source_repo": source_repo,
    "source_revision": source_revision,
    "model_name": os.environ["MODEL_NAME"],
    "quant_type": quant_type,
    "intermediate_outtype": os.environ["INTERMEDIATE_OUTTYPE"],
    "convert_split_max_size": os.environ["CONVERT_SPLIT_MAX_SIZE"],
    "tensor_type_overrides": os.environ.get("TENSOR_TYPE_OVERRIDES", ""),
    "keep_output_tensor": os.environ.get("KEEP_OUTPUT_TENSOR", ""),
    "token_embedding_type": os.environ.get("TOKEN_EMBEDDING_TYPE", ""),
    "files": files,
}
manifest_path = quant_dir / "jianyang-quant-manifest.json"
manifest_path.write_text(json.dumps(manifest, indent=2))
api.upload_file(
    repo_id=repo,
    repo_type="model",
    path_or_fileobj=str(manifest_path),
    path_in_repo="jianyang-quant-manifest.json",
    commit_message="Add Jianyang quant manifest",
)

info = api.model_info(repo, repo_type="model")
Path(os.environ["GGUF_REVISION_FILE"]).write_text(info.sha or "main")
PY

TARGET_GGUF_REVISION="$(cat "$GGUF_REVISION_FILE")"
echo "Target GGUF revision: $TARGET_GGUF_REVISION"

log_step "Prepare /source view for layer package job"
SOURCE_FILE="${GGUF_SUBDIR}/$(basename "$QUANT_FIRST")"
mkdir -p "/source/${GGUF_SUBDIR}"
find "$QUANT_DIR" -maxdepth 1 -type f -name '*.gguf' -print0 |
    while IFS= read -r -d '' shard; do
        ln -sf "$shard" "/source/${GGUF_SUBDIR}/$(basename "$shard")"
    done

log_step "Run Skippy layer package publisher"
export SOURCE_REPO="$TARGET_GGUF_REPO"
export SOURCE_REVISION="$TARGET_GGUF_REVISION"
export SOURCE_FILE
export SOURCE_QUANT="$QUANT_TYPE"
SOURCE_TOTAL_BYTES="$(find "$QUANT_DIR" -maxdepth 1 -type f -name '*.gguf' -print0 |
    xargs -0 stat --format='%s' 2>/dev/null |
    awk '{sum += $1} END {print sum + 0}')"
export SOURCE_TOTAL_BYTES
export TARGET_REPO="$TARGET_PACKAGE_REPO"
export MODEL_ID="$MODEL_ID"
export MESH_LLM_REF="$MESH_LLM_REF"
export CATALOG_CREATE_PR="${CATALOG_CREATE_PR:-false}"
export JOB_WORK_ROOT="${JOB_WORK_ROOT:-${WORK_DIR}/package-job-work}"
export LOCAL_WORK_DIR="${LOCAL_WORK_DIR:-${WORK_DIR}/package-local-work}"

bash "$PACKAGE_SCRIPT"

log_step "Phase 1 job complete"
echo "GGUF repo:    https://huggingface.co/${TARGET_GGUF_REPO}"
echo "Package repo: https://huggingface.co/${TARGET_PACKAGE_REPO}"
echo "Quant bytes:  $(format_bytes "$SOURCE_TOTAL_BYTES")"
