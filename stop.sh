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

for name in \
  "${container_prefix}-k2-tp1" \
  "${container_prefix}-k2-tp2-head" "${container_prefix}-native-tp2-head"; do
  remote "${head_host}" docker rm -f "${name}" >/dev/null 2>&1 || true
done
for name in "${container_prefix}-k2-tp2-worker" "${container_prefix}-native-tp2-worker"; do
  remote "${worker_host}" docker rm -f "${name}" >/dev/null 2>&1 || true
done
echo "Stopped ds4-mia containers."
