#!/usr/bin/env python3
"""Install an opt-in two-pass K3 DSpark experiment.

The normal DSpark implementation emits all configured draft tokens in one
parallel block.  Vision-Exp is strongest at K3, while positions four and later
collapse in a one-shot K6 block.  This hotfix adds two *experimental* K3+K3
paths without changing the default implementation:

``DSPARK_RECURSIVE_MODE=kv``
    Preserve the first K3 pass's per-layer query KV and make it context for a
    second K3 pass, anchored by the third sampled token.

``DSPARK_RECURSIVE_MODE=tap``
    Also reconstruct the mHC output after every draft layer, mean-pool the four
    hyper-connection lanes, concatenate the three 4096-wide taps, apply the
    checkpoint's existing ``main_proj``/``main_norm``, and store the resulting
    target-shaped states as context KV before the second pass.

Both modes require ``num_speculative_tokens=6`` and eager execution.  With the
environment variable unset, the appended code takes no runtime branch and the
original K3 production path is byte-for-byte intact above the injection.
"""
from __future__ import annotations

import sys
from pathlib import Path


DEFAULT_MODEL = Path(
    "/usr/local/lib/python3.12/dist-packages/vllm/models/deepseek_v4/nvidia/dspark.py"
)
DEFAULT_SPECULATOR = Path(
    "/usr/local/lib/python3.12/dist-packages/vllm/v1/worker/gpu/spec_decode/"
    "dspark/speculator.py"
)

MODEL_MARK = "# [recursive-dspark-hotfix] expose target-shaped draft taps"
SPECULATOR_MARK = "# [recursive-dspark-hotfix] two-pass K3 experiment"


MODEL_INJECT = r'''

# [recursive-dspark-hotfix] expose target-shaped draft taps
import os as _recursive_dspark_os

_recursive_dspark_original_model_forward = DSparkDeepseekV4Model.forward


def _recursive_dspark_model_forward(
    self,
    input_ids: torch.Tensor,
    positions: torch.Tensor,
    inputs_embeds: torch.Tensor | None = None,
) -> torch.Tensor:
    mode = _recursive_dspark_os.environ.get("DSPARK_RECURSIVE_MODE", "")
    if mode != "tap":
        return _recursive_dspark_original_model_forward(
            self, input_ids, positions, inputs_embeds
        )

    if inputs_embeds is None:
        inputs_embeds = self.embed_input_ids(input_ids)
    hidden_states = inputs_embeds.unsqueeze(-2).repeat(1, self.hc_mult, 1)

    residual = post_mix = res_mix = None
    taps = []
    reconstructed = None
    for layer in self.layers:
        hidden_states, residual, post_mix, res_mix = layer(
            hidden_states,
            positions,
            input_ids,
            post_mix,
            res_mix,
            residual,
        )
        # This is the same boundary and the same hc-lane mean used by the
        # target model for dspark_target_layer_ids.  mhc_post allocates an
        # output and does not mutate the delayed state carried to the next layer.
        reconstructed = mhc_post_tilelang(
            hidden_states, residual, post_mix, res_mix
        )
        taps.append(reconstructed.mean(dim=-2))

    if len(taps) != len(self.target_layer_ids):
        raise RuntimeError(
            "recursive DSpark tap count mismatch: "
            f"draft={len(taps)} target={len(self.target_layer_ids)}"
        )
    self._recursive_context_states = self.combine_hidden_states(
        torch.cat(taps, dim=-1)
    )
    assert reconstructed is not None
    return hc_head_fused_kernel_tilelang(
        reconstructed,
        self.hc_head_fn,
        self.hc_head_scale,
        self.hc_head_base,
        self.rms_norm_eps,
        self.hc_eps,
    )


DSparkDeepseekV4Model.forward = _recursive_dspark_model_forward
'''


SPECULATOR_INJECT = r'''

# [recursive-dspark-hotfix] two-pass K3 experiment
import os as _recursive_dspark_os
from vllm.v1.worker.gpu.attn_utils import (
    build_slot_mappings_by_layer as _recursive_build_slot_mappings_by_layer,
)
from vllm.v1.worker.gpu.dp_utils import (
    dispatch_cg_and_sync_dp as _recursive_dispatch_cg_and_sync_dp,
)
from vllm.v1.worker.gpu.spec_decode.dflash.speculator import (
    prepare_dflash_inputs as _recursive_prepare_dflash_inputs,
)

_recursive_dspark_original_init = DSparkSpeculator.__init__


def _recursive_dspark_init(self, vllm_config, device):
    _recursive_dspark_original_init(self, vllm_config, device)
    mode = _recursive_dspark_os.environ.get("DSPARK_RECURSIVE_MODE", "")
    self._recursive_dspark_mode = mode
    if not mode:
        return
    if mode not in ("kv", "tap"):
        raise ValueError(
            "DSPARK_RECURSIVE_MODE must be empty, 'kv', or 'tap'; "
            f"got {mode!r}"
        )
    if self.num_speculative_steps != 6:
        raise ValueError(
            "recursive DSpark requires num_speculative_tokens=6; "
            f"got {self.num_speculative_steps}"
        )
    if not self.sample_from_anchor:
        raise ValueError("recursive DSpark requires the anchor-as-first layout")

    # The scheduler/verifier still sees K6.  Each draft forward sees only K3.
    self.num_query_per_req = 3
    self._anchor_idx = (
        torch.arange(self.max_num_reqs, dtype=torch.int64, device=device) * 3
    )


def _recursive_dspark_sample_block(
    self,
    num_reqs: int,
    head_hidden: torch.Tensor,
    output_col: int,
) -> None:
    width = 3
    sample_hidden = head_hidden[: num_reqs * width].view(
        num_reqs, width, head_hidden.shape[-1]
    )
    base_logits = self.model.compute_draft_logits(sample_hidden)
    vocab_size = base_logits.shape[-1]
    base_logits = base_logits.view(num_reqs, width, vocab_size)
    positions = self.input_buffers.positions[: num_reqs * width].view(
        num_reqs, width
    )
    idx_map = self.idx_mapping[:num_reqs]
    inputs = self.input_buffers.input_ids[: num_reqs * width].view(num_reqs, width)
    prev = inputs[:, 0]

    for i in range(width):
        markov_embed = self.model.markov_embed(prev)
        logits_i = base_logits[:, i] + self.model.markov_bias(markov_embed)
        if self.draft_logits is not None:
            if self._d2t_scatter_index is not None:
                assert self._draft_scatter_buf is not None
                buf = self._draft_scatter_buf[:num_reqs]
                buf.fill_(float("-inf"))
                buf.index_copy_(1, self._d2t_scatter_index, logits_i.to(buf.dtype))
                logits_i = buf
            # DSpark's Gumbel key is the predecessor position.  The query row
            # at P predicts P+1, so this is P (matching the stock implementation).
            sampled = gumbel_sample(
                logits_i,
                idx_map,
                self.temperature,
                self.seeds,
                positions[:, i],
                apply_temperature=True,
                output_processed_logits=self.draft_logits,
                output_processed_logits_col=self._step_cols[output_col + i],
                use_fp64=self.use_fp64_gumbel,
            )
        else:
            sampled = self.model.map_draft_to_target(logits_i.argmax(dim=-1))
        self.draft_tokens[:num_reqs, output_col + i] = sampled
        prev = sampled


def _recursive_dspark_prepare_second_block(self, num_reqs: int) -> None:
    width = 3
    num_tokens = num_reqs * width
    positions = self.input_buffers.positions[:num_tokens].view(num_reqs, width)
    base = positions[:, 0].clone()
    offsets = torch.arange(width, dtype=positions.dtype, device=positions.device)
    positions.copy_(
        torch.clamp(
            base[:, None] + width + offsets[None, :],
            max=self.max_model_len - 1,
        )
    )

    input_ids = self.input_buffers.input_ids[:num_tokens].view(num_reqs, width)
    input_ids.fill_(self.parallel_drafting_token_id)
    input_ids[:, 0].copy_(self.draft_tokens[:num_reqs, width - 1])

    qsl = self.input_buffers.query_start_loc
    qsl[: num_reqs + 1].copy_(
        torch.arange(
            num_reqs + 1, dtype=qsl.dtype, device=qsl.device
        ) * width
    )
    qsl[num_reqs + 1 :].fill_(num_tokens)
    seq_lens = self.input_buffers.seq_lens
    seq_lens[:num_reqs].copy_(
        torch.clamp(base + 2 * width, max=self.max_model_len)
    )
    seq_lens[num_reqs:].zero_()


def _recursive_dspark_tap_feedback(
    self,
    num_reqs: int,
    num_query_tokens: int,
) -> None:
    if self._recursive_dspark_mode != "tap":
        return
    draft_model = self.model.model
    states = getattr(draft_model, "_recursive_context_states", None)
    if states is None:
        raise RuntimeError("recursive DSpark tap states were not produced")
    states = states[:num_query_tokens]
    positions = self.input_buffers.positions[:num_query_tokens]

    if self._layer_group_idx is None:
        slots = self.block_tables.slot_mappings[
            self.draft_kv_cache_group_id, :num_query_tokens
        ]
    else:
        slots = [
            self.block_tables.slot_mappings[
                self.draft_kv_cache_group_ids[gidx], :num_query_tokens
            ]
            for gidx in self._layer_group_idx
        ]
    self.model.precompute_and_store_context_kv(states, positions, slots)


@torch.inference_mode()
def _recursive_dspark_propose(
    self,
    input_batch,
    attn_metadata,
    slot_mappings,
    last_hidden_states,
    aux_hidden_states,
    num_sampled,
    num_rejected,
    last_sampled,
    next_prefill_tokens,
    temperature,
    seeds,
    num_tokens_across_dp=None,
    dummy_run=False,
    skip_attn_for_dummy_run=False,
    mm_inputs=None,
    is_profile=False,
):
    if not self._recursive_dspark_mode:
        return DFlashSpeculator.propose(
            self,
            input_batch,
            attn_metadata,
            slot_mappings,
            last_hidden_states,
            aux_hidden_states,
            num_sampled,
            num_rejected,
            last_sampled,
            next_prefill_tokens,
            temperature,
            seeds,
            num_tokens_across_dp,
            dummy_run,
            skip_attn_for_dummy_run,
            mm_inputs,
            is_profile,
        )

    width = 3
    num_reqs = input_batch.num_reqs
    num_target_tokens = input_batch.num_tokens
    num_query_tokens = num_reqs * width
    max_seq_len = input_batch.seq_lens_cpu_upper_bound[:num_reqs].max().item()
    self.draft_max_seq_len = min(max_seq_len + 2 * width, self.max_model_len)

    if aux_hidden_states:
        hidden_states = self.model.combine_hidden_states(
            torch.cat(aux_hidden_states, dim=-1)
        )
    else:
        hidden_states = last_hidden_states
    self.hidden_states[:num_target_tokens].copy_(hidden_states[:num_target_tokens])
    self._copy_request_inputs(num_reqs, input_batch.idx_mapping, temperature, seeds)

    if dummy_run and skip_attn_for_dummy_run:
        # Only shapes and peak allocations matter during the memory profile.
        self.model.precompute_and_store_context_kv(
            self.hidden_states[:num_target_tokens],
            self.context_positions[:num_target_tokens],
        )
        self.input_buffers.input_ids[:num_query_tokens].fill_(
            self.parallel_drafting_token_id
        )
        self.input_buffers.positions[:num_query_tokens].zero_()
        self._prepare_eplb_forward(num_query_tokens)
        first = self._run_model(
            num_query_tokens, None, None, num_tokens_across_dp, CUDAGraphMode.NONE
        )
        _recursive_dspark_sample_block(self, num_reqs, first, 0)
        _recursive_dspark_prepare_second_block(self, num_reqs)
        second = self._run_model(
            num_query_tokens, None, None, num_tokens_across_dp, CUDAGraphMode.NONE
        )
        _recursive_dspark_sample_block(self, num_reqs, second, width)
        return self.draft_tokens[:num_reqs]

    assert self.draft_kv_cache_group_id >= 0
    for i, gid in enumerate(self.draft_kv_cache_group_ids):
        _recursive_prepare_dflash_inputs(
            self.input_buffers,
            self.block_tables.slot_mappings[gid],
            self.context_positions,
            self._context_slot_mappings[i],
            self.sample_indices,
            self.sample_pos,
            self.sample_idx_mapping,
            input_batch,
            num_sampled,
            num_rejected,
            last_sampled,
            next_prefill_tokens,
            self.block_tables.input_block_tables[gid],
            self.block_tables.block_sizes[gid],
            self.parallel_drafting_token_id,
            width,
            6,
            self.max_num_reqs,
            self.max_num_tokens,
            self.max_model_len,
            True,
        )

    if dummy_run:
        context_slots = None
    elif self._layer_group_idx is not None:
        context_slots = [
            self._context_slot_mappings[gidx][:num_target_tokens]
            for gidx in self._layer_group_idx
        ]
    else:
        context_slots = self._context_slot_mappings[0][:num_target_tokens]
    self.model.precompute_and_store_context_kv(
        self.hidden_states[:num_target_tokens],
        self.context_positions[:num_target_tokens],
        context_slots,
    )

    batch_desc, num_tokens_across_dp = _recursive_dispatch_cg_and_sync_dp(
        self.query_cudagraph_manager,
        num_reqs,
        num_query_tokens,
        uniform_token_count=width,
        dp_size=self.dp_size,
        dp_rank=self.dp_rank,
        need_eager=True,
    )
    if batch_desc.cg_mode != CUDAGraphMode.NONE:
        raise RuntimeError("recursive DSpark currently requires --enforce-eager")
    num_reqs_padded = batch_desc.num_reqs or num_reqs
    num_tokens_padded = batch_desc.num_tokens
    if num_tokens_padded != num_query_tokens or num_reqs_padded != num_reqs:
        raise RuntimeError(
            "recursive DSpark eager batch was unexpectedly padded: "
            f"requests={num_reqs_padded}/{num_reqs}, "
            f"tokens={num_tokens_padded}/{num_query_tokens}"
        )

    def run_block(output_col):
        draft_attn_metadata = self._build_draft_attn_metadata(
            num_reqs=num_reqs,
            num_reqs_padded=num_reqs,
            num_tokens_padded=num_query_tokens,
            causal=self.dflash_causal,
        )
        draft_slots = _recursive_build_slot_mappings_by_layer(
            self.block_tables.slot_mappings[:, :num_query_tokens],
            self.kv_cache_config,
        )
        self._prepare_eplb_forward(num_query_tokens)
        head_hidden = self._run_model(
            num_query_tokens,
            draft_attn_metadata,
            draft_slots,
            num_tokens_across_dp,
            CUDAGraphMode.NONE,
        )
        _recursive_dspark_sample_block(self, num_reqs, head_hidden, output_col)

    run_block(0)
    _recursive_dspark_tap_feedback(self, num_reqs, num_query_tokens)

    _recursive_dspark_prepare_second_block(self, num_reqs)
    self.block_tables.compute_slot_mappings(
        self.idx_mapping[:num_reqs],
        self.input_buffers.query_start_loc[: num_reqs + 1],
        self.input_buffers.positions[:num_query_tokens],
        num_query_tokens,
    )
    run_block(width)
    return self.draft_tokens[:num_reqs]


DSparkSpeculator.__init__ = _recursive_dspark_init
DSparkSpeculator.propose = _recursive_dspark_propose
'''


def patch_text(source: str, marker: str, injection: str, required: str) -> tuple[str, str]:
    if marker in source:
        return source, "skipped"
    if required not in source:
        return source, f"drift:missing-{required}"
    updated = source.rstrip() + "\n" + injection
    compile(updated, "recursive-dspark-hotfix.py", "exec")
    return updated, "applied"


def _write(path: Path, original: str, updated: str, status: str) -> None:
    if status == "applied":
        path.write_text(updated)
    elif status != "skipped":
        raise SystemExit(f"FATAL: {path} {status}")
    print(f"recursive DSpark hotfix {path.name:30s}: {status}")


def main() -> int:
    if len(sys.argv) > 1 and sys.argv[1] == "--status":
        model = DEFAULT_MODEL.read_text() if DEFAULT_MODEL.is_file() else ""
        spec = DEFAULT_SPECULATOR.read_text() if DEFAULT_SPECULATOR.is_file() else ""
        print("recursive DSpark model     :", "APPLIED" if MODEL_MARK in model else "NOT APPLIED")
        print(
            "recursive DSpark speculator:",
            "APPLIED" if SPECULATOR_MARK in spec else "NOT APPLIED",
        )
        return 0 if MODEL_MARK in model and SPECULATOR_MARK in spec else 1

    model_path = Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_MODEL
    spec_path = Path(sys.argv[2]) if len(sys.argv) > 2 else DEFAULT_SPECULATOR
    originals = {model_path: model_path.read_text(), spec_path: spec_path.read_text()}
    model_updated, model_status = patch_text(
        originals[model_path], MODEL_MARK, MODEL_INJECT, "class DSparkDeepseekV4Model"
    )
    spec_updated, spec_status = patch_text(
        originals[spec_path], SPECULATOR_MARK, SPECULATOR_INJECT, "class DSparkSpeculator"
    )
    # Validate both before modifying either: this is an atomic source-lock gate.
    if model_status.startswith("drift:") or spec_status.startswith("drift:"):
        raise SystemExit(
            f"FATAL: model={model_status}, speculator={spec_status}"
        )
    _write(model_path, originals[model_path], model_updated, model_status)
    _write(spec_path, originals[spec_path], spec_updated, spec_status)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
