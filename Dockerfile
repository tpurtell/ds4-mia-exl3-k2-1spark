# syntax=docker/dockerfile:1.7
# Mia's current production lane: vLLM 0.25.2.dev plus the selected DSV4
# hotfixes backported from vLLM 0.27. The EXL3 image is only a source stage;
# none of its runtime is inherited by the final image.
ARG EXL3_SOURCE_IMAGE=ghcr.io/tpurtell/deepseek-v4-flash-0731-exl3-k2-spark@sha256:bf383b32a03bdcfef19e42b52778df413c0c47d07c3f4d4e66c78002d17beb74
ARG MIA_BASE_IMAGE=ghcr.io/anemll/dspark-vllm-gx10@sha256:a83948492cf13df455170fb42885f5ef4db54fefe0feff0f841ecbff464ac9d8

FROM ${EXL3_SOURCE_IMAGE} AS exl3_source
FROM ${MIA_BASE_IMAGE}

ARG B12X_REPOSITORY=https://github.com/tpurtell/sparkinfer-glmrt.git
ARG B12X_COMMIT=28e083482fd18ca3ce0e2553cd533102be85552f

SHELL ["/bin/bash", "-c"]

# Keep Mia's vLLM, CUDA, FlashInfer, torch, and DSpark stack intact. Only put
# the current b12x tree ahead of the older package shipped by the base image.
# Mia's production image intentionally omits git, so fetch the immutable source
# archive with Python instead of installing another package manager layer.
RUN B12X_REPOSITORY="${B12X_REPOSITORY}" B12X_COMMIT="${B12X_COMMIT}" \
    python3 - <<'PY'
import os
import shutil
import tarfile
import urllib.request
from pathlib import Path

repository = os.environ["B12X_REPOSITORY"].removesuffix(".git")
commit = os.environ["B12X_COMMIT"]
archive = Path("/tmp/b12x.tar.gz")
urllib.request.urlretrieve(f"{repository}/archive/{commit}.tar.gz", archive)
for stale in Path("/tmp").glob("sparkinfer-glmrt-*"):
    shutil.rmtree(stale)
with tarfile.open(archive) as tar:
    tar.extractall("/tmp", filter="data")
sources = list(Path("/tmp").glob("sparkinfer-glmrt-*"))
if len(sources) != 1:
    raise RuntimeError(f"expected one b12x source tree, found: {sources}")
source = sources[0]
# Mia's native MXFP4 integration imports the legacy b12x.moe.fused and
# integration.tp_moe namespaces. The current tree adds fused_moe but no longer
# ships those modules, so carry forward only files absent from the new tree.
legacy = Path("/opt/b12x/b12x")
current = source / "b12x"
if legacy.is_dir():
    for old in legacy.rglob("*"):
        relative = old.relative_to(legacy)
        if "__pycache__" in relative.parts:
            continue
        new = current / relative
        if old.is_dir():
            new.mkdir(parents=True, exist_ok=True)
        elif not new.exists():
            new.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(old, new)
shutil.rmtree("/opt/b12x", ignore_errors=True)
shutil.move(source, "/opt/b12x")
integration_init = Path("/opt/b12x/b12x/integration/__init__.py")
export = "from .tp_moe import prepare_b12x_fp4_moe_weights"
text = integration_init.read_text()
if export not in text:
    integration_init.write_text(text.rstrip() + f"\n\n{export}\n")
archive.unlink()
PY
RUN python3 -m pip install --no-deps -e /opt/b12x

# The normal safetensors path stages checkpoint pages in host memory before
# copying into EXL3's compact GPU slabs. On unified-memory GB10 that briefly
# accounts both copies against the same 128 GiB pool. InstantTensor streams one
# tensor at a time directly to CUDA, which is the loader used by our qualified
# standard EXL3 image and avoids that load-time peak without changing Mia's
# execution runtime.
RUN python3 -m pip install --no-deps 'instanttensor==0.1.5'
COPY patches/port-instanttensor.py /tmp/port-instanttensor.py
RUN python3 /tmp/port-instanttensor.py \
    /usr/local/lib/python3.12/dist-packages/instanttensor/_impl.py

ENV PYTHONPATH=/opt/b12x:/usr/local/lib/python3.12/dist-packages

COPY --from=exl3_source \
    /opt/vllm/vllm/model_executor/layers/quantization/exl3.py \
    /usr/local/lib/python3.12/dist-packages/vllm/model_executor/layers/quantization/exl3.py
COPY patches/port-exl3-mia.py /tmp/port-exl3-mia.py
RUN python3 /tmp/port-exl3-mia.py \
    /usr/local/lib/python3.12/dist-packages/vllm
COPY patches/port-exl3-mixed.py /tmp/port-exl3-mixed.py
RUN python3 /tmp/port-exl3-mixed.py \
    /usr/local/lib/python3.12/dist-packages/vllm

# Mia's current recipe applies these selected vLLM 0.27 DSV4 backports at
# launch. Bake the same idempotent patches into the public image instead.
COPY patches/hotfix-dsv4-skip-topk-49486.sh \
     patches/hotfix-dsv4-mtp-buffer-50312.sh \
     patches/hotfix-dsv4-skip-empty-c128-48957.sh \
     patches/hotfix-dsv4-flashmla-workspace-50298.sh \
     patches/hotfix-dsv4-dense-prefill-indexer-48407.sh \
     patches/hotfix-dsv4-grammar-advance.sh \
     patches/hotfix-gb10-spin-wait.sh \
     patches/hotfix-vllm-redact-api-key-log.sh \
     /tmp/mia-hotfixes/
COPY patches/hotfix-nvfp4-ds-mla-issue22.sh /tmp/mia-hotfixes/hotfix-nvfp4-ds-mla-issue22.sh
RUN for patch in /tmp/mia-hotfixes/*.sh; do \
      VLLM_ROOT=/usr/local/lib/python3.12/dist-packages/vllm bash "${patch}"; \
    done

# Keep the one-Spark image on the same mandatory Python hotfix set as the
# current upstream recipe. Optional behavior-changing patches stay available
# to the entrypoint and remain opt-in.
COPY patches/hotfix-encoding-dsv4-issue21.py \
     patches/port-dsv4-reasoning-effort.py \
     patches/hotfix-vllm-safetensors-index.py \
     patches/hotfix-dsv4-issue55-tool-truncation.py \
     patches/hotfix-vllm-empty-encoder-output.py \
     patches/hotfix-dsv4-issue27-partial-prefill-concurrency.py \
     patches/hotfix-dsv4-issue43-decode-fairness-and-diag.py \
     patches/hotfix-dsv4-issue26-hybrid-swa-min.py \
     patches/hotfix-dsv4-suppress-stops-in-reasoning.py \
     patches/hotfix-dsv4-issue31-v2-thinking-budget-gpu.py \
     patches/hotfix-dsv4-assistant-final-continuation.py \
     /opt/recipe/patches/
RUN python3 /opt/recipe/patches/port-dsv4-reasoning-effort.py && \
    python3 /opt/recipe/patches/hotfix-encoding-dsv4-issue21.py && \
    python3 /opt/recipe/patches/hotfix-vllm-safetensors-index.py \
      /usr/local/lib/python3.12/dist-packages/vllm && \
    python3 /opt/recipe/patches/hotfix-dsv4-issue55-tool-truncation.py \
      /usr/local/lib/python3.12/dist-packages/vllm && \
    python3 /opt/recipe/patches/hotfix-vllm-empty-encoder-output.py && \
    python3 /opt/recipe/patches/hotfix-dsv4-issue27-partial-prefill-concurrency.py && \
    python3 /opt/recipe/patches/hotfix-dsv4-issue43-decode-fairness-and-diag.py && \
    python3 /opt/recipe/patches/hotfix-dsv4-issue26-hybrid-swa-min.py && \
    python3 /opt/recipe/patches/hotfix-dsv4-suppress-stops-in-reasoning.py

RUN python3 - <<'PY'
from pathlib import Path

import b12x
import instanttensor
import torch
import vllm
from importlib import import_module
from b12x.integration import prepare_b12x_fp4_moe_weights
from vllm.model_executor.layers.quantization import get_quantization_config
from vllm.model_executor.layers.quantization.exl3 import Exl3Config

assert vllm.__version__.startswith("0.25.2.dev"), vllm.__version__
assert torch.__version__.startswith("2.11."), torch.__version__
assert Path(b12x.__file__).is_relative_to(Path("/opt/b12x")), b12x.__file__
assert Path(instanttensor.__file__).is_file(), instanttensor.__file__
assert import_module("b12x.moe.fused_moe") is not None
assert import_module("b12x.integration.tp_moe") is not None
assert import_module("b12x.moe.fused.w4a16.host") is not None
assert callable(prepare_b12x_fp4_moe_weights)
assert get_quantization_config("exl3") is Exl3Config
assert Exl3Config({"bits": 2, "quant_method": "exl3"}).get_name() == "exl3"
flashmla = Path(vllm.__file__).parent / "v1/attention/backends/mla/flashmla_sparse.py"
assert 'self.kv_cache_dtype in ("fp8_ds_mla", "nvfp4_ds_mla")' in flashmla.read_text()
structured = Path(vllm.__file__).parent / "v1/structured_output/__init__.py"
scheduler = Path(vllm.__file__).parent / "v1/core/sched/scheduler.py"
assert "new_token_ids: list[int] | None = None" in structured.read_text()
assert "should_advance(\n                request, new_token_ids=new_token_ids\n            )" in scheduler.read_text()
print("Mia runtime + EXL3 port verified")
PY

COPY scripts /opt/recipe/scripts
RUN chmod 0755 /opt/recipe/scripts/*.sh

LABEL org.opencontainers.image.source="https://github.com/tpurtell/ds4-mia-exl3-k2-1spark" \
      org.opencontainers.image.description="Mia DeepSeek V4 Flash DSpark runtime with EXL3 K2 and mixed K2/K3 support" \
      org.opencontainers.image.licenses="MIT"

EXPOSE 8888
HEALTHCHECK --interval=30s --timeout=10s --start-period=30m --retries=10 \
  CMD curl -fsS "http://127.0.0.1:${VLLM_PORT:-8888}/health" >/dev/null || exit 1
ENTRYPOINT ["/opt/recipe/scripts/k2-entrypoint.sh"]
