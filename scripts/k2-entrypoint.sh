#!/usr/bin/env bash
set -Eeuo pipefail

model_kind=${MODEL_KIND:-k2}
case "${model_kind}" in
  k2|k2-v1)
    model_repo=${MODEL_REPO:-wrldsuksgo2mars/DeepSeek-V4-Flash-0731-EXL3-K2-calibrated-v1}
    model_revision=${MODEL_REVISION:-68eaca43e99bfbfd697a5559c7796b983deb38f8}
    served_model_name=${SERVED_MODEL_NAME:-deepseek-v4-flash-0731-exl3-k2-calibrated-v1}
    ;;
  k2-v0)
    model_repo=${MODEL_REPO:-wrldsuksgo2mars/DeepSeek-V4-Flash-0731-EXL3-K2-calibrated-v0}
    model_revision=${MODEL_REVISION:-dff9afc6f5fe50a890590f7b6d5339ceaf5ba51e}
    served_model_name=${SERVED_MODEL_NAME:-deepseek-v4-flash-0731-exl3-k2-calibrated-v0}
    ;;
  k21|k21-v2)
    model_repo=${MODEL_REPO:-wrldsuksgo2mars/DeepSeek-V4-Flash-0731-EXL3-K2.1-calibrated-v2}
    model_revision=${MODEL_REVISION:-a2b066719ebdc0cbb0eacc752ffe7a2190c919aa}
    served_model_name=${SERVED_MODEL_NAME:-deepseek-v4-flash-0731-exl3-k2.1-calibrated-v2}
    ;;
  k21-v1)
    model_repo=${MODEL_REPO:-wrldsuksgo2mars/DeepSeek-V4-Flash-0731-EXL3-K2.1-calibrated-v1}
    model_revision=${MODEL_REVISION:-73757f619a951d812fe8008a39dbade8df20e6c6}
    served_model_name=${SERVED_MODEL_NAME:-deepseek-v4-flash-0731-exl3-k2.1-calibrated-v1}
    ;;
  native)
    model_repo=${MODEL_REPO:-deepseek-ai/DeepSeek-V4-Flash-0731}
    model_revision=${MODEL_REVISION:-9e165c30e2704aec5d9d593cce3eebd58bbef1cb}
    served_model_name=${SERVED_MODEL_NAME:-deepseek-v4-flash-0731-native}
    ;;
  *)
    echo "MODEL_KIND must be k2, k2-v0, k2-v1, k21, k21-v1, k21-v2, or native; got '${model_kind}'" >&2
    exit 2
    ;;
esac

model_ref=${MODEL_PATH:-${model_repo}}
revision_args=()
if [[ -n "${model_revision}" && -z "${MODEL_PATH:-}" ]]; then
  revision_args=(--revision "${model_revision}")
fi

tp_size=${TP_SIZE:-1}
nnodes=${NNODES:-1}
node_rank=${NODE_RANK:-0}
if [[ ! "${tp_size}" =~ ^[1-9][0-9]*$ ]]; then
  echo "TP_SIZE must be a positive integer; got '${tp_size}'" >&2
  exit 2
fi
if [[ ! "${nnodes}" =~ ^[1-9][0-9]*$ ]]; then
  echo "NNODES must be a positive integer; got '${nnodes}'" >&2
  exit 2
fi

distributed_args=()
if (( nnodes > 1 )); then
  : "${MASTER_ADDR:?MASTER_ADDR is required for multi-node launch}"
  distributed_args=(
    --nnodes "${nnodes}"
    --node-rank "${node_rank}"
    --master-addr "${MASTER_ADDR}"
    --master-port "${MASTER_PORT:-25000}"
  )
  if (( node_rank > 0 )); then
    distributed_args+=(--headless)
  fi
fi

executor_backend=${DISTRIBUTED_EXECUTOR_BACKEND:-}
if [[ -z "${executor_backend}" ]]; then
  if (( tp_size == 1 && nnodes == 1 )); then
    executor_backend=uni
  else
    executor_backend=mp
  fi
fi
case "${executor_backend}" in
  uni|mp|ray|external_launcher) ;;
  *)
    echo "DISTRIBUTED_EXECUTOR_BACKEND must be uni, mp, ray, or external_launcher; got '${executor_backend}'" >&2
    exit 2
    ;;
esac

prefix_args=()
if [[ "${PREFIX_CACHE:-1}" == 1 ]]; then
  prefix_args=(--enable-prefix-caching)
fi

thinking=${DEFAULT_THINKING:-max}
case "${thinking}" in
  off) chat_kwargs='{"thinking":false}' ;;
  low|high|max) chat_kwargs="{\"thinking\":true,\"reasoning_effort\":\"${thinking}\"}" ;;
  *) echo "DEFAULT_THINKING must be off, low, high, or max" >&2; exit 2 ;;
esac

draft_sample_method=${DRAFT_SAMPLE_METHOD:-probabilistic}
case "${draft_sample_method}" in
  probabilistic|greedy) ;;
  *)
    echo "DRAFT_SAMPLE_METHOD must be probabilistic or greedy; got '${draft_sample_method}'" >&2
    exit 2
    ;;
esac
speculative_config=$(printf \
  '{"method":"dspark","num_speculative_tokens":%s,"draft_sample_method":"%s"}' \
  "${DSPARK_TOKENS:-5}" "${draft_sample_method}")

api_key_args=()
case "${DSPARK_API_KEYS:-}" in
  *[$'\r\n\v\f']*)
    echo "DSPARK_API_KEYS must be a single-line space-separated list" >&2
    exit 2
    ;;
  *\\*)
    echo "DSPARK_API_KEYS must not contain backslashes" >&2
    exit 2
    ;;
esac
dspark_keys_set=0
case "${DSPARK_API_KEYS:-}" in *[!$' \t']*) dspark_keys_set=1 ;; esac
if [[ -n "${VLLM_API_KEY:-}" && "${dspark_keys_set}" == 1 ]]; then
  echo "VLLM_API_KEY and DSPARK_API_KEYS are both set; set exactly one" >&2
  exit 2
fi
if [[ "${dspark_keys_set}" == 1 ]]; then
  read -r -a dspark_keys <<< "${DSPARK_API_KEYS}"
  for dspark_key in "${dspark_keys[@]}"; do
    case "${dspark_key}" in
      -*) echo "DSPARK_API_KEYS contains a token beginning with '-'" >&2; exit 2 ;;
    esac
  done
  api_key_args=(--api-key "${dspark_keys[@]}")
fi

case "${DSPARK_ENABLE_ISSUE31_GPU_HOTFIX:-0}" in
  0) ;;
  1) python3 /opt/recipe/patches/hotfix-dsv4-issue31-v2-thinking-budget-gpu.py ;;
  *) echo "DSPARK_ENABLE_ISSUE31_GPU_HOTFIX must be 0 or 1" >&2; exit 2 ;;
esac
case "${DSPARK_ENABLE_ASSISTANT_FINAL_HOTFIX:-0}" in
  0) ;;
  1) python3 /opt/recipe/patches/hotfix-dsv4-assistant-final-continuation.py ;;
  *) echo "DSPARK_ENABLE_ASSISTANT_FINAL_HOTFIX must be 0 or 1" >&2; exit 2 ;;
esac

export HF_HOME=${HF_HOME:-/cache/huggingface}
export HF_HUB_OFFLINE=${HF_HUB_OFFLINE:-1}
export HF_HUB_DISABLE_XET=${HF_HUB_DISABLE_XET:-1}
export TRANSFORMERS_OFFLINE=${TRANSFORMERS_OFFLINE:-1}
export VLLM_CACHE_ROOT=${VLLM_CACHE_ROOT:-/cache/huggingface/vllm-cache}
export CUTE_DSL_ARCH=${CUTE_DSL_ARCH:-sm_121a}
export TORCH_CUDA_ARCH_LIST=${TORCH_CUDA_ARCH_LIST:-12.1a}
export FLASHINFER_CUDA_ARCH_LIST=${FLASHINFER_CUDA_ARCH_LIST:-12.1a}
export FLASHINFER_DISABLE_VERSION_CHECK=${FLASHINFER_DISABLE_VERSION_CHECK:-1}
export FLASHINFER_WORKSPACE_BASE=${FLASHINFER_WORKSPACE_BASE:-/cache/huggingface/flashinfer}
export TILELANG_CLEANUP_TEMP_FILES=${TILELANG_CLEANUP_TEMP_FILES:-1}
export DG_JIT_USE_NVRTC=${DG_JIT_USE_NVRTC:-0}
export DG_JIT_NVCC_COMPILER=${DG_JIT_NVCC_COMPILER:-/usr/local/cuda/bin/nvcc}
export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}
export VLLM_ALLOW_LONG_MAX_MODEL_LEN=${VLLM_ALLOW_LONG_MAX_MODEL_LEN:-1}
export VLLM_SPARSE_INDEXER_MAX_LOGITS_MB=${VLLM_SPARSE_INDEXER_MAX_LOGITS_MB:-256}
export VLLM_MEMORY_PROFILER_ESTIMATE_CUDAGRAPHS=${VLLM_MEMORY_PROFILER_ESTIMATE_CUDAGRAPHS:-0}
export VLLM_USE_BREAKABLE_CUDAGRAPH=${VLLM_USE_BREAKABLE_CUDAGRAPH:-0}
export VLLM_USE_FLASHINFER_SAMPLER=${VLLM_USE_FLASHINFER_SAMPLER:-1}
export VLLM_PREFIX_CACHE_RETENTION_INTERVAL=${VLLM_PREFIX_CACHE_RETENTION_INTERVAL:-4096}
export DSPARK_MAX_INFLIGHT_PREFILLS=${DSPARK_MAX_INFLIGHT_PREFILLS:-2}
export DSPARK_ISSUE43_SCHED_DIAG=${DSPARK_ISSUE43_SCHED_DIAG:-0}
export VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS=${VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS:-1800}
export TILELANG_CACHE_DIR=${TILELANG_CACHE_DIR:-/cache/huggingface/tilelang-cache}
export TRITON_CACHE_DIR=${TRITON_CACHE_DIR:-/cache/huggingface/triton-cache}
export B12X_COMPILE_CACHE_DIR=${B12X_COMPILE_CACHE_DIR:-/cache/huggingface/b12x-compile-cache}
export VLLM_USE_B12X_MOE=${VLLM_USE_B12X_MOE:-1}
export VLLM_B12X_W4A16_FORCE_BLOCKS_PER_SM=${VLLM_B12X_W4A16_FORCE_BLOCKS_PER_SM:-0}
export VLLM_B12X_W4A16_FORCE_BLOCKS_MAX_M=${VLLM_B12X_W4A16_FORCE_BLOCKS_MAX_M:-16}
export NCCL_NET=${NCCL_NET:-IB}
export NCCL_DEBUG=${NCCL_DEBUG:-WARN}
for optional_nccl_env in \
  NCCL_IB_MERGE_NICS NCCL_NET_GDR_LEVEL NCCL_NET_GDR_READ NCCL_DMABUF_ENABLE; do
  if [[ -z "${!optional_nccl_env:-}" ]]; then
    unset "${optional_nccl_env}"
  fi
done

exec /usr/local/bin/vllm serve "${model_ref}" \
  "${revision_args[@]}" \
  --served-model-name "${served_model_name}" \
  --host "${VLLM_HOST:-0.0.0.0}" \
  --port "${VLLM_PORT:-8888}" \
  --trust-remote-code \
  --tensor-parallel-size "${tp_size}" \
  --pipeline-parallel-size 1 \
  --kv-cache-dtype "${KV_CACHE_DTYPE:-nvfp4_ds_mla}" \
  --block-size 256 \
  --max-model-len "${MAX_MODEL_LEN:-1000000}" \
  --max-num-seqs "${MAX_NUM_SEQS:-6}" \
  --max-num-batched-tokens "${MAX_NUM_BATCHED_TOKENS:-8192}" \
  --long-prefill-token-threshold "${LONG_PREFILL_TOKEN_THRESHOLD:-1024}" \
  --max-cudagraph-capture-size "${MAX_CUDAGRAPH_CAPTURE_SIZE:-36}" \
  --gpu-memory-utilization "${GPU_MEMORY_UTILIZATION:-0.85}" \
  --load-format "${LOAD_FORMAT:-instanttensor}" \
  "${prefix_args[@]}" \
  --enable-prompt-tokens-details \
  --async-scheduling \
  --enable-chunked-prefill \
  --speculative-config "${speculative_config}" \
  --tokenizer-mode deepseek_v4 \
  --distributed-executor-backend "${executor_backend}" \
  --moe-backend flashinfer_b12x \
  --tool-call-parser deepseek_v4 \
  --enable-auto-tool-choice \
  --reasoning-parser deepseek_v4 \
  --reasoning-config '{"reasoning_parser":"deepseek_v4","reasoning_start_str":"<think>","reasoning_end_str":"</think>"}' \
  --default-chat-template-kwargs "${chat_kwargs}" \
  --generation-config vllm \
  --enable-flashinfer-autotune \
  "${api_key_args[@]}" \
  "${distributed_args[@]}" \
  "$@"
