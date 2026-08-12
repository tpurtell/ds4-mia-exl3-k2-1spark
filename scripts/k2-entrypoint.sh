#!/usr/bin/env bash
set -Eeuo pipefail

model_kind=${MODEL_KIND:-k2}
case "${model_kind}" in
  k2)
    model_repo=${MODEL_REPO:-wrldsuksgo2mars/DeepSeek-V4-Flash-0731-EXL3-K2-calibrated-v1}
    model_revision=${MODEL_REVISION:-68eaca43e99bfbfd697a5559c7796b983deb38f8}
    served_model_name=${SERVED_MODEL_NAME:-deepseek-v4-flash-0731-exl3-k2-calibrated-v1}
    ;;
  native)
    model_repo=${MODEL_REPO:-deepseek-ai/DeepSeek-V4-Flash-0731}
    model_revision=${MODEL_REVISION:-9e165c30e2704aec5d9d593cce3eebd58bbef1cb}
    served_model_name=${SERVED_MODEL_NAME:-deepseek-v4-flash-0731-native}
    ;;
  *)
    echo "MODEL_KIND must be k2 or native; got '${model_kind}'" >&2
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

speculative_config=$(printf \
  '{"method":"dspark","num_speculative_tokens":%s,"draft_sample_method":"%s"}' \
  "${DSPARK_TOKENS:-5}" "${DRAFT_SAMPLE_METHOD:-probabilistic}")

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
export VLLM_USE_B12X_MOE=${VLLM_USE_B12X_MOE:-1}
export VLLM_B12X_W4A16_FORCE_BLOCKS_PER_SM=${VLLM_B12X_W4A16_FORCE_BLOCKS_PER_SM:-0}
export VLLM_B12X_W4A16_FORCE_BLOCKS_MAX_M=${VLLM_B12X_W4A16_FORCE_BLOCKS_MAX_M:-16}
export NCCL_NET=${NCCL_NET:-IB}
export NCCL_DEBUG=${NCCL_DEBUG:-WARN}

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
  --max-cudagraph-capture-size "${MAX_CUDAGRAPH_CAPTURE_SIZE:-36}" \
  --gpu-memory-utilization "${GPU_MEMORY_UTILIZATION:-0.85}" \
  --load-format "${LOAD_FORMAT:-instanttensor}" \
  "${prefix_args[@]}" \
  --enable-prompt-tokens-details \
  --async-scheduling \
  --enable-chunked-prefill \
  --speculative-config "${speculative_config}" \
  --tokenizer-mode deepseek_v4 \
  --distributed-executor-backend mp \
  --moe-backend flashinfer_b12x \
  --tool-call-parser deepseek_v4 \
  --enable-auto-tool-choice \
  --reasoning-parser deepseek_v4 \
  --reasoning-config '{"reasoning_parser":"deepseek_v4","reasoning_start_str":"<think>","reasoning_end_str":"</think>"}' \
  --default-chat-template-kwargs "${chat_kwargs}" \
  --generation-config vllm \
  --enable-flashinfer-autotune \
  "${distributed_args[@]}" \
  "$@"
