#!/usr/bin/env bash
set -Eeuo pipefail

head_host=${HEAD_HOST:-localhost}
worker_host=${WORKER_HOST:-dodo}
container_prefix=${CONTAINER_PREFIX:-ds4-mia}

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

model_kinds=(
  k2 k2-v0 k2-v1
  vision-k2 vision vision-k22 vision-k2.2 vision-k22-d2
  k21-d22 k2.1-d2.2 k21 k21-v1 k21-v2 native
)
for model_kind in "${model_kinds[@]}"; do
  remote "${head_host}" docker rm -f \
    "${container_prefix}-${model_kind}-tp1" \
    "${container_prefix}-${model_kind}-tp2-head" >/dev/null 2>&1 || true
  remote "${worker_host}" docker rm -f \
    "${container_prefix}-${model_kind}-tp2-worker" >/dev/null 2>&1 || true
done
echo "Stopped ds4-mia containers."
