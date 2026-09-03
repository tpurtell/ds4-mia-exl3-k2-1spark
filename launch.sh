#!/usr/bin/env bash
set -Eeuo pipefail

script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
env_file=${ENV_FILE:-${script_dir}/.env}
if [[ -f "${env_file}" ]]; then
  set -a
  # shellcheck disable=SC1090
  source "${env_file}"
  set +a
fi

nodes=1
model_kind=${MODEL_KIND:-vision-k22}
while (( $# )); do
  case "$1" in
    --nodes) nodes=$2; shift 2 ;;
    --model) model_kind=$2; shift 2 ;;
    -h|--help)
      echo "usage: $0 [--nodes 1|2] [--model k2|k2-v0|k2-v1|vision-k2|vision-k22|k21-d22|k21|k21-v1|k21-v2|native]"
      exit 0
      ;;
    *) echo "unknown argument: $1" >&2; exit 2 ;;
  esac
done
case "${nodes}" in 1|2) ;; *) echo "--nodes must be 1 or 2" >&2; exit 2 ;; esac
case "${model_kind}" in
  k2|k2-v0|k2-v1|vision-k2|vision|vision-k22|vision-k2.2|vision-k22-d2|k21-d22|k2.1-d2.2|k21|k21-v1|k21-v2|native) ;;
  *) echo "unsupported --model '${model_kind}'" >&2; exit 2 ;;
esac
if (( nodes == 1 )) && [[ "${model_kind}" == native ]]; then
  echo "The native checkpoint needs two Sparks; use --nodes 2." >&2
  exit 2
fi

image=${RECIPE_IMAGE:-ghcr.io/tpurtell/ds4-mia-exl3-k2-1spark:latest}
hf_cache=${HF_CACHE:-${HOME}/.cache/huggingface}
container_prefix=${CONTAINER_PREFIX:-ds4-mia}
head_host=${HEAD_HOST:-localhost}
worker_host=${WORKER_HOST:-dodo}
head_ip=${HEAD_IP:-127.0.0.1}
worker_ip=${WORKER_IP:-10.55.0.2}
head_nccl_hca=${NCCL_IB_HCA:-=rocep1s0f0,roceP2p1s0f0}
worker_nccl_hca=${WORKER_NCCL_IB_HCA:-${head_nccl_hca}}
head_nccl_if=${NCCL_SOCKET_IFNAME:-enp1s0f0np0}
worker_nccl_if=${WORKER_NCCL_SOCKET_IFNAME:-${head_nccl_if}}
head_gid=${NCCL_IB_GID_INDEX:-}
worker_gid=${WORKER_NCCL_IB_GID_INDEX:-}

remote() {
  local host=$1 command
  shift
  printf -v command '%q ' "$@"
  if [[ "${host}" == localhost || "${host}" == "$(hostname)" ]]; then
    bash -lc "${command}"
  else
    ssh -o BatchMode=yes "${host}" "${command}"
  fi
}

resolve_rocev2_gid_index() {
  local host=$1 ifname=$2 hca_list=$3 script
  printf -v script 'ifname=%q; hca_list=%q; ' "${ifname}" "${hca_list}"
  script+='hca_list=${hca_list#=}; indexes=""; IFS=, read -ra hcas <<<"$hca_list"; '
  script+='for hca in "${hcas[@]}"; do found=""; for type_file in /sys/class/infiniband/$hca/ports/1/gid_attrs/types/*; do '
  script+='[[ $(cat "$type_file" 2>/dev/null) == "RoCE v2" ]] || continue; index=${type_file##*/}; '
  script+='gid=$(cat /sys/class/infiniband/$hca/ports/1/gids/$index 2>/dev/null); '
  script+='[[ $gid == *ffff:* ]] && { found=$index; break; }; done; '
  script+='[[ -n $found ]] || { echo "cannot resolve IPv4 RoCEv2 GID for $hca" >&2; exit 1; }; indexes+=" $found"; done; '
  script+='set -- $indexes; first=$1; for index in "$@"; do [[ $index == "$first" ]] || { echo "HCAs require different GID indexes: $indexes" >&2; exit 1; }; done; echo "$first"'
  remote "${host}" bash -lc "${script}"
}

if (( nodes == 2 )) && [[ "${NCCL_IB_GID_AUTO:-1}" == 1 ]]; then
  head_gid=$(resolve_rocev2_gid_index "${head_host}" "${head_nccl_if}" "${head_nccl_hca}")
  worker_gid=$(resolve_rocev2_gid_index "${worker_host}" "${worker_nccl_if}" "${worker_nccl_hca}")
  echo "RoCEv2 GID index: ${head_host}=${head_gid}, ${worker_host}=${worker_gid}"
fi

common_args=(
  --gpus all --network host --ipc host --shm-size 64g
  --device /dev/infiniband:/dev/infiniband
  --ulimit memlock=-1:-1 --ulimit nofile=1048576:1048576
  --ulimit stack=67108864:67108864
  -v "${hf_cache}:/cache/huggingface"
  -e MODEL_KIND="${model_kind}"
  -e MODEL_REPO="${MODEL_REPO:-}"
  -e MODEL_REVISION="${MODEL_REVISION:-}"
  -e SERVED_MODEL_NAME="${SERVED_MODEL_NAME:-}"
  -e DSPARK_ENCODING_FILE="${DSPARK_ENCODING_FILE:-}"
  -e HF_HOME=/cache/huggingface
  -e HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
  -e HF_HUB_DISABLE_XET="${HF_HUB_DISABLE_XET:-1}"
  -e TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"
  -e VLLM_CACHE_ROOT=/cache/huggingface/vllm-cache
  -e MAX_MODEL_LEN="${MAX_MODEL_LEN:-1000000}"
  -e MAX_NUM_SEQS="${MAX_NUM_SEQS:-6}"
  -e MAX_NUM_BATCHED_TOKENS="${MAX_NUM_BATCHED_TOKENS:-8192}"
  -e LONG_PREFILL_TOKEN_THRESHOLD="${LONG_PREFILL_TOKEN_THRESHOLD:-1024}"
  -e MAX_CUDAGRAPH_CAPTURE_SIZE="${MAX_CUDAGRAPH_CAPTURE_SIZE:-}"
  -e GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-}"
  -e KV_CACHE_DTYPE="${KV_CACHE_DTYPE:-nvfp4_ds_mla}"
  -e PREFIX_CACHE="${PREFIX_CACHE:-1}"
  -e DSPARK_ENFORCE_EAGER="${DSPARK_ENFORCE_EAGER:-0}"
  -e DSPARK_TOKENS="${DSPARK_TOKENS:-}"
  -e LIMIT_MM_PER_PROMPT="${LIMIT_MM_PER_PROMPT:-image=8}"
  -e DRAFT_SAMPLE_METHOD="${DRAFT_SAMPLE_METHOD:-probabilistic}"
  -e DEFAULT_THINKING="${DEFAULT_THINKING:-max}"
  -e DSPARK_MAX_INFLIGHT_PREFILLS="${DSPARK_MAX_INFLIGHT_PREFILLS:-2}"
  -e DSPARK_ISSUE43_SCHED_DIAG="${DSPARK_ISSUE43_SCHED_DIAG:-0}"
  -e VLLM_PREFIX_CACHE_RETENTION_INTERVAL="${VLLM_PREFIX_CACHE_RETENTION_INTERVAL:-4096}"
  -e DSPARK_ENABLE_ISSUE31_GPU_HOTFIX="${DSPARK_ENABLE_ISSUE31_GPU_HOTFIX:-0}"
  -e DSPARK_ENABLE_ASSISTANT_FINAL_HOTFIX="${DSPARK_ENABLE_ASSISTANT_FINAL_HOTFIX:-0}"
  -e DSPARK_ENABLE_ISSUE136_XGRAMMAR_HOTFIX="${DSPARK_ENABLE_ISSUE136_XGRAMMAR_HOTFIX:-1}"
  -e DSPARK_SUPPRESS_STOPS_IN_REASONING="${DSPARK_SUPPRESS_STOPS_IN_REASONING:-1}"
  -e VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS="${VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS:-1800}"
  -e TILELANG_CACHE_DIR="${TILELANG_CACHE_DIR:-/cache/huggingface/tilelang-cache}"
  -e TRITON_CACHE_DIR="${TRITON_CACHE_DIR:-/cache/huggingface/triton-cache}"
  -e B12X_COMPILE_CACHE_DIR="${B12X_COMPILE_CACHE_DIR:-/cache/huggingface/b12x-compile-cache}"
  -e DISTRIBUTED_EXECUTOR_BACKEND="${DISTRIBUTED_EXECUTOR_BACKEND:-}"
  -e CUTE_DSL_ARCH=sm_121a
  -e VLLM_ALLOW_LONG_MAX_MODEL_LEN=1
  -e VLLM_SPARSE_INDEXER_MAX_LOGITS_MB="${VLLM_SPARSE_INDEXER_MAX_LOGITS_MB:-256}"
  -e VLLM_MEMORY_PROFILER_ESTIMATE_CUDAGRAPHS=0
  -e VLLM_USE_B12X_MOE=1
  -e VLLM_B12X_W4A16_FORCE_BLOCKS_PER_SM="${VLLM_B12X_W4A16_FORCE_BLOCKS_PER_SM:-0}"
  -e VLLM_B12X_W4A16_FORCE_BLOCKS_MAX_M="${VLLM_B12X_W4A16_FORCE_BLOCKS_MAX_M:-16}"
  -e NCCL_IB_DISABLE="${NCCL_IB_DISABLE:-0}"
  -e NCCL_NET="${NCCL_NET:-IB}"
  -e NCCL_DEBUG="${NCCL_DEBUG:-WARN}"
  -e NCCL_IB_ADDR_FAMILY="${NCCL_IB_ADDR_FAMILY:-AF_INET}"
  -e NCCL_IB_ROCE_VERSION_NUM="${NCCL_IB_ROCE_VERSION_NUM:-2}"
  -e NCCL_CROSS_NIC="${NCCL_CROSS_NIC:-1}"
  -e NCCL_CUMEM_ENABLE="${NCCL_CUMEM_ENABLE:-0}"
  -e NCCL_IGNORE_CPU_AFFINITY="${NCCL_IGNORE_CPU_AFFINITY:-1}"
  -e NCCL_NVLS_ENABLE="${NCCL_NVLS_ENABLE:-0}"
)

if [[ -n "${VLLM_API_KEY:-}" ]]; then
  common_args+=(-e VLLM_API_KEY="${VLLM_API_KEY}")
fi
if [[ -n "${DSPARK_API_KEYS:-}" ]]; then
  common_args+=(-e DSPARK_API_KEYS="${DSPARK_API_KEYS}")
fi
for optional_nccl_env in \
  NCCL_IB_MERGE_NICS NCCL_NET_GDR_LEVEL NCCL_NET_GDR_READ NCCL_DMABUF_ENABLE; do
  if [[ -n "${!optional_nccl_env:-}" ]]; then
    common_args+=(-e "${optional_nccl_env}=${!optional_nccl_env}")
  fi
done

if (( nodes == 1 )); then
  name=${container_prefix}-${model_kind}-tp1
  remote "${head_host}" docker rm -f "${name}" >/dev/null 2>&1 || true
  remote "${head_host}" docker run -d --name "${name}" "${common_args[@]}" \
    -e TP_SIZE=1 -e NNODES=1 "${image}" >/dev/null
  echo "Started ${head_host}/${name}; follow with: ssh ${head_host} docker logs -f ${name}"
  exit 0
fi

head_name=${container_prefix}-${model_kind}-tp2-head
worker_name=${container_prefix}-${model_kind}-tp2-worker
remote "${head_host}" docker rm -f "${head_name}" >/dev/null 2>&1 || true
remote "${worker_host}" docker rm -f "${worker_name}" >/dev/null 2>&1 || true

remote "${worker_host}" docker run -d --name "${worker_name}" \
  "${common_args[@]}" -e TP_SIZE=2 -e NNODES=2 -e NODE_RANK=1 \
  -e NCCL_IB_HCA="${worker_nccl_hca}" -e NCCL_SOCKET_IFNAME="${worker_nccl_if}" \
  -e TP_SOCKET_IFNAME="${WORKER_TP_SOCKET_IFNAME:-${worker_nccl_if}}" \
  -e GLOO_SOCKET_IFNAME="${WORKER_GLOO_SOCKET_IFNAME:-${worker_nccl_if}}" \
  -e NCCL_IB_GID_INDEX="${worker_gid}" \
  -e VLLM_HOST_IP="${worker_ip}" -e MASTER_ADDR="${head_ip}" \
  -e MASTER_PORT="${MASTER_PORT:-25000}" "${image}" >/dev/null
remote "${head_host}" docker run -d --name "${head_name}" \
  "${common_args[@]}" -e TP_SIZE=2 -e NNODES=2 -e NODE_RANK=0 \
  -e NCCL_IB_HCA="${head_nccl_hca}" -e NCCL_SOCKET_IFNAME="${head_nccl_if}" \
  -e TP_SOCKET_IFNAME="${TP_SOCKET_IFNAME:-${head_nccl_if}}" \
  -e GLOO_SOCKET_IFNAME="${GLOO_SOCKET_IFNAME:-${head_nccl_if}}" \
  -e NCCL_IB_GID_INDEX="${head_gid}" \
  -e VLLM_HOST_IP="${head_ip}" -e MASTER_ADDR="${head_ip}" \
  -e MASTER_PORT="${MASTER_PORT:-25000}" "${image}" >/dev/null

echo "Started ${model_kind} TP2. Head: ${head_host}/${head_name}; worker: ${worker_host}/${worker_name}"
