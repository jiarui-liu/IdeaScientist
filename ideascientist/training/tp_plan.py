"""Tensor-parallel plan and numerics patches for Qwen3.5 under NeMo-RL.

nemo_automodel does not register ``Qwen3_5ForConditionalGeneration``, and the
model's HF ``_tp_plan`` uses a parallel style the translator does not recognise,
so parallelization silently falls back to the llama-style base plan. That plan
keys its transformer entries on ``model.layers.*`` while qwen3.5 nests them
under ``model.language_model.layers.*``, so the only key it matches is
``lm_head`` — which it assigns ``ColwiseParallel(output_layouts=Replicate)``.
A replicated ``lm_head`` all-gathers the full vocabulary (248320) over the whole
generated sequence at logprob time, which does not fit.

:func:`qwen3_5_tp_plan` therefore vocab-shards ``lm_head`` and shards the MLP.
Sharding ``lm_head`` is safe for the colocated-vLLM refit: the send path calls
``full_tensor`` on every DTensor before streaming and the vocabulary divides
evenly by TP=8, so vLLM receives the same ``[vocab, hidden]`` tensor either way.

The attention and linear-attention blocks stay replicated. qwen3.5 is a hybrid
GatedDeltaNet / full-attention model and upstream ships no plan for the
linear_attn projections.

The rest of this module installs patches against NeMo-RL internals, each one
guarded by an environment variable so it can be turned off against a version
that no longer needs it. The one that is not optional in practice is
:func:`_install_fp32_params_refit_key_patch`: the trainer wraps the fp32 SSM
parameters in a ``_fp32_params`` holder, so their streamed key carries an infix
that vLLM's flat ``linear_attn.A_log`` / ``dt_bias`` do not match. The weights
are then skipped in silence and left at vLLM's random init, which destroys the
recurrent state while leaving generation fluent — the failure looks like a
model that never terminates, not like a load error.
"""
from __future__ import annotations

import os

from torch.distributed.tensor import Shard
from torch.distributed.tensor.parallel import ColwiseParallel, RowwiseParallel


_LINEAR_ATTN_FWD_PATCH_INSTALLED = False








# --- Chunk the train-loss vocab-parallel logprob backward ---------------------
# The DTensor train-loss backward materializes two full-vocab [BS_local, V]
# buffers at once, and the transient grows with trajectory length, so a run that
# is learning reaches a memory ceiling it did not hit at the start. There is no
# config lever — logprob_chunk_size gates the no-grad pass, not the train loss —
# so substitute the framework's chunked variant when the caller passes no chunk.
_LOGPROB_CHUNK = int(os.environ.get("IDEASCIENTIST_LOGPROB_CHUNK", "2048"))
_PATCH_INSTALLED = False


def _install_chunked_logprob_patch() -> None:
    global _PATCH_INSTALLED
    if _PATCH_INSTALLED or _LOGPROB_CHUNK <= 0:
        return
    from nemo_rl.distributed import model_utils as _mu

    _orig = _mu.get_logprobs_from_vocab_parallel_logits

    def _chunked(vocab_parallel_logits, input_ids, seq_index=None,
                 chunk_size=None, sampling_params=None):
        if chunk_size is None:
            chunk_size = _LOGPROB_CHUNK
        return _orig(
            vocab_parallel_logits,
            input_ids,
            seq_index=seq_index,
            chunk_size=chunk_size,
            sampling_params=sampling_params,
        )

    _mu.get_logprobs_from_vocab_parallel_logits = _chunked
    _PATCH_INSTALLED = True
    print(
        f"[tp_plan] patched get_logprobs_from_vocab_parallel_logits "
        f"to chunk the train-loss vocab-parallel backward (chunk_size={_LOGPROB_CHUNK})"
    )


# Install on import: this module is imported inside each DTensorPolicyWorkerV2 via
# get_class(custom_parallel_plan) (parallelize.py), i.e. in the exact process
# where the train backward runs — so the patch lands where it is needed.
_install_chunked_logprob_patch


# --- In-place concat of the chunked vocab-parallel backward gradient ----------
# The chunked backward accumulates per-chunk gradients into a list and joins them
# with one torch.cat at the end, so the full list and the cat output are both
# resident at that instant — twice the final gradient. Chunk size does not help:
# the cat output is the full [BS, V_local] whatever the chunk. Preallocate the
# output and copy each chunk into its row slice instead. The staticmethod is
# replaced wholesale because a single line cannot be monkeypatched; the values are
# bit-identical. Disable via IDEASCIENTIST_INPLACE_CONCAT=0.
_INPLACE_CONCAT = os.environ.get("IDEASCIENTIST_INPLACE_CONCAT", "1") != "0"
_INPLACE_CONCAT_PATCH_INSTALLED = False


def _install_inplace_concat_patch() -> None:
    global _INPLACE_CONCAT_PATCH_INSTALLED
    if _INPLACE_CONCAT_PATCH_INSTALLED or not _INPLACE_CONCAT:
        return
    import torch
    from nemo_rl.distributed import model_utils as _mu

    _Fn = _mu.ChunkedDistributedLogprobWithSampling

    @staticmethod
    def _backward_inplace(ctx, *grad_outputs):
        # Faithful reimplementation of ChunkedDistributedLogprobWithSampling.backward
        # (model_utils.py). ONLY change: preallocate the [BS, V_local] output
        # and slice-copy each chunk into it (dropping the chunk) instead of appending
        # to a list and torch.cat-ing at the end. Values are bit-identical.
        grad_output = grad_outputs[0]  # [B, S]
        (vocab_parallel_logits,) = ctx.saved_tensors
        target = ctx.target
        tp_group = ctx.tp_group
        top_k = ctx.top_k
        top_p = ctx.top_p
        chunk_size = ctx.chunk_size

        world_size = torch.distributed.get_world_size(tp_group)
        rank = torch.distributed.get_rank(tp_group)
        B, S, V_local = vocab_parallel_logits.shape
        BS = B * S

        effective_chunk_size = chunk_size * B
        reshaped_vocab_parallel_logits = vocab_parallel_logits.view(BS, V_local)
        target_flat = target.flatten

        num_chunks = (BS + effective_chunk_size - 1) // effective_chunk_size
        grad_output_flat = grad_output.flatten

        # Preallocate the full output ONCE (replaces all_grad_chunks list + torch.cat).
        grad_vocab_parallel = torch.empty(
            (BS, V_local),
            dtype=vocab_parallel_logits.dtype,
            device=vocab_parallel_logits.device,
        )

        for chunk_idx in range(num_chunks):
            chunk_start = chunk_idx * effective_chunk_size
            chunk_end = min(BS, (chunk_idx + 1) * effective_chunk_size)
            current_chunk_size = chunk_end - chunk_start
            local_chunk_size = current_chunk_size // world_size

            vocab_parallel_logits_chunk = reshaped_vocab_parallel_logits[
                chunk_start:chunk_end, :
            ]

            target_chunk = target_flat[chunk_start:chunk_end]
            target_local = target_chunk[
                rank * local_chunk_size : (rank + 1) * local_chunk_size
            ]

            seq_parallel_logits_chunk = _mu.all_to_all_vp2sq(
                vocab_parallel_logits_chunk, tp_group
            )

            logits_chunk, keep_mask = _mu.apply_top_k_top_p(
                seq_parallel_logits_chunk, top_k=top_k, top_p=top_p
            )

            log_probs_chunk = torch.nn.functional.log_softmax(
                logits_chunk.to(dtype=torch.float32), dim=-1
            )
            softmax_chunk = log_probs_chunk.exp

            grad_chunk = grad_output_flat[chunk_start:chunk_end]
            grad_local = grad_chunk[
                rank * local_chunk_size : (rank + 1) * local_chunk_size
            ]

            V = softmax_chunk.shape[-1]
            is_chosen = torch.nn.functional.one_hot(target_local, num_classes=V)
            grad_logits_local = is_chosen.float.sub_(softmax_chunk)
            grad_logits_local.mul_(grad_local.unsqueeze(-1))

            if keep_mask is not None:
                grad_logits_local.mul_(keep_mask)

            grad_vocab_parallel_chunk = _mu.all_to_all_sq2vp(
                grad_logits_local, tp_group
            )
            # Copy straight into the preallocated row-slice, then drop the chunk so its
            # memory frees before the next iteration (this is what removes the ~4.76 GiB
            # list-plus-cat double-allocation peak).
            grad_vocab_parallel[chunk_start:chunk_end].copy_(grad_vocab_parallel_chunk)
            del grad_vocab_parallel_chunk

        grad_vocab_parallel = grad_vocab_parallel.view(B, S, V_local)

        return grad_vocab_parallel, None, None, None, None, None, None

    _Fn.backward = _backward_inplace
    _INPLACE_CONCAT_PATCH_INSTALLED = True
    print(
        "[tp_plan] patched ChunkedDistributedLogprobWithSampling.backward "
        "to preallocate the [BS,V_local] grad and slice-copy each chunk (drops the "
        "list+torch.cat double-allocation ~9.5GiB peak to ~4.76GiB; bit-identical)"
    )


_install_inplace_concat_patch


# --- CPU-offload the per-layer activation-checkpoint boundary residuals -------
# A custom_parallel_plan supplied as an import path is returned verbatim, so the
# parallelizer never applies sequence-parallel wrapping and ``sequence_parallel``
# in the config is a silent no-op. The per-layer checkpoint boundary residuals are
# therefore replicated on every TP rank and held from forward through backward,
# scaling with sequence length. Real sequence-parallelism is not available here
# (the GatedDeltaNet recurrence is not sequence-shardable), so the only lever that
# preserves sequence length is to page those boundaries to CPU: run the checkpoint
# call inside save_on_cpu, which under the reentrant path intercepts the region
# inputs only — recompute happens later, outside the context, and stays on GPU.
# Disable via IDEASCIENTIST_AC_OFFLOAD=0.
_AC_OFFLOAD = os.environ.get("IDEASCIENTIST_AC_OFFLOAD", "1") != "0"
_AC_OFFLOAD_PIN = os.environ.get("IDEASCIENTIST_AC_OFFLOAD_PIN", "0") != "0"
_AC_PATCH_INSTALLED = False


def _install_activation_offload_patch() -> None:
    global _AC_PATCH_INSTALLED
    if _AC_PATCH_INSTALLED or not _AC_OFFLOAD:
        return
    import torch
    from transformers import modeling_utils as _mu

    _orig_ckpt = _mu.checkpoint

    def _offloading_checkpoint(function, *args, **kwargs):
        # Page the checkpoint region's saved-for-backward tensors (the [1,S,H]
        # layer-boundary residual) to CPU. kwargs (incl. use_reentrant=True) pass
        # through untouched, so the AC semantics are identical to steps 1-7.
        with torch.autograd.graph.save_on_cpu(pin_memory=_AC_OFFLOAD_PIN):
            return _orig_ckpt(function, *args, **kwargs)

    _mu.checkpoint = _offloading_checkpoint
    _AC_PATCH_INSTALLED = True
    print(
        "[tp_plan] patched transformers.modeling_utils.checkpoint to "
        f"CPU-offload activation-checkpoint boundary residuals (pin_memory={_AC_OFFLOAD_PIN})"
    )


_install_activation_offload_patch


# --- Allow partial optimizer-state resume -------------------------------------
# LoRA is applied to both towers of the VLM base, but this pipeline is text-only,
# so the visual tower never receives gradients and lazy Adam never materializes
# its state: the saved optimizer state covers the language model alone. On resume
# the destination materializes zero-state for every trainable param, so it expects
# keys the checkpoint does not have and the strict default load planner raises —
# surfacing as an unrelated pickle error, because the real exception is wrapped
# with a traceback that cannot be pickled for the cross-rank gather.
#
# Skipping those keys is correct: the visual params are untrained, and fresh zero
# Adam state is exactly what a from-scratch optimizer would hold. Relax only the
# optimizer load, identified by its top-level keys; the model load takes a
# different path and other dcp.load calls stay strict.
# Disable via IDEASCIENTIST_PARTIAL_OPTIM_LOAD=0.
_PARTIAL_OPTIM_LOAD = os.environ.get("IDEASCIENTIST_PARTIAL_OPTIM_LOAD", "1") != "0"
_PARTIAL_LOAD_PATCH_INSTALLED = False




def _install_partial_optim_load_patch() -> None:
    global _PARTIAL_LOAD_PATCH_INSTALLED
    if _PARTIAL_LOAD_PATCH_INSTALLED or not _PARTIAL_OPTIM_LOAD:
        return
    import torch.distributed.checkpoint as _dcp
    from torch.distributed.checkpoint.default_planner import DefaultLoadPlanner

    _orig_load = _dcp.load

    def _partial_optim_load(state_dict, *args, **kwargs):
        # Only relax the OPTIMIZER load: its top-level keys are a subset of
        # {"optim","sched"}. Leave every other dcp.load (incl. any strict model
        # load) untouched, and never override an explicitly supplied planner.
        is_optim = (
            isinstance(state_dict, dict)
            and len(state_dict) > 0
            and set(state_dict.keys) <= {"optim", "sched"}
        )
        return _orig_load(state_dict, *args, **kwargs)

    _dcp.load = _partial_optim_load
    _PARTIAL_LOAD_PATCH_INSTALLED = True
    print(
        "[tp_plan] patched torch.distributed.checkpoint.load to allow "
        "partial optimizer-state resume (untrained visual-tower params)"
    )


_install_partial_optim_load_patch


# --- Strip the `_fp32_params` infix from streamed refit keys ------------------
# qwen3.5's two SSM parameters, A_log and dt_bias, must stay fp32, so nemo_automodel
# relocates them into a ``_fp32_params`` holder submodule to wrap them in their own
# FSDP group. ``state_dict`` then emits them with the holder infix, while vLLM
# stores them flat — so the streamed key misses params_dict and is skipped in
# silence, leaving both at vLLM's random init. Random decay and timestep in every
# linear-attention layer destroys the recurrent state tracking while leaving
# per-token fluency intact, which looks like a model that never terminates rather
# than like a load failure.
#
# Strip the infix on the send side. Both refit paths go through the same symbol, so
# the registered and streamed key sets stay in lockstep and the byte-offset
# assertion still holds — this is a rename, not a value change.
# Disable via IDEASCIENTIST_FP32_REFIT_KEY_FIX=0.
_FP32_REFIT_KEY_FIX = os.environ.get("IDEASCIENTIST_FP32_REFIT_KEY_FIX", "1") != "0"
_FP32_KEY_PATCH_INSTALLED = False





def _install_fp32_params_refit_key_patch() -> None:
    global _FP32_KEY_PATCH_INSTALLED
    if _FP32_KEY_PATCH_INSTALLED or not _FP32_REFIT_KEY_FIX:
        return
    # The worker module defines this class, so it is already imported in this
    # process by the time the custom plan is loaded during parallelization.
    from nemo_rl.models.policy.workers import dtensor_policy_worker_v2 as _w

    _orig_adapt = _w._maybe_adapt_tensor_to_hf

    def _adapt_strip_fp32(model_part, fqn, tensor, quantization=False):
        out = _orig_adapt(model_part, fqn, tensor, quantization=quantization)
        out = [
            (k.replace("._fp32_params.", ".") if "._fp32_params." in k else k, t)
            for k, t in out
        ]
        # The trainer is text-only Qwen3_5ForCausalLM but the colocated vLLM
        # engine keeps the VLM architecture, whose language weights live under
        # `language_model.*`. Without the prefix, refit streams `model.*` and vLLM
        # rejects it. The guard makes this a no-op for a VLM trainer, and it
        # affects the refit path only, so on-disk keys stay resume-safe.
        out = [
            (
                k
                if (k.startswith("language_model.") or k.startswith("visual."))
                else "language_model." + k,
                t,
            )
            for k, t in out
        ]
        return out

    _w._maybe_adapt_tensor_to_hf = _adapt_strip_fp32
    _FP32_KEY_PATCH_INSTALLED = True
    print(
        "[tp_plan] patched dtensor_policy_worker_v2._maybe_adapt_tensor_to_hf "
        "to strip the '_fp32_params' holder infix from streamed refit keys so vLLM's "
        "flat linear_attn.A_log/dt_bias params load (were silently left at dummy init)"
    )


_install_fp32_params_refit_key_patch


# --- Load the fp32 SSM holder params from the base checkpoint ----------------
# The send-side rename above is necessary but not sufficient: the stream carried
# the right key with a zero value, because the trainer's holder parameter was
# never filled from disk. For TP>1 the base checkpoint loads after parallelize,
# by which time the live FQN already carries the holder infix, and the state-dict
# adapter has no infix handling — so the request key never matches disk's flat
# key, the value is dropped under strict=False, and the holder keeps its init.
#
# Rename in both directions around the adapter: strip the infix on the way out so
# DCP locates the disk value, re-add it on the way back so load_state_dict writes
# into the holder. Disable via IDEASCIENTIST_FP32_HOLDER_LOAD_FIX=0.
_FP32_HOLDER_LOAD_FIX = os.environ.get("IDEASCIENTIST_FP32_HOLDER_LOAD_FIX", "1") != "0"
_FP32_HOLDER_LOAD_PATCH_INSTALLED = False
_FP32_HOLDER_INFIX = "._fp32_params."
# Only these two GatedDeltaNet params live in the fp32 holder (cp_linear_attn.py).
_FP32_HOLDER_PARAMS = ("A_log", "dt_bias")


def _install_fp32_holder_load_patch() -> None:
    global _FP32_HOLDER_LOAD_PATCH_INSTALLED
    if _FP32_HOLDER_LOAD_PATCH_INSTALLED or not _FP32_HOLDER_LOAD_FIX:
        return
    # Standalone import (no model needed); patching the CLASS covers any adapter
    # instance created at model-build time, since instance methods resolve via the class.
    from nemo_automodel.components.models.qwen3_5_moe.state_dict_adapter import (
        Qwen3_5MoeStateDictAdapter as _Adapter,
    )

    _orig_convert = _Adapter.convert_single_tensor_to_hf
    _orig_from_hf = _Adapter.from_hf

    def _convert_strip_holder(self, fqn, tensor, **kwargs):
        out = _orig_convert(self, fqn, tensor, **kwargs)
        return [
            ((k.replace(_FP32_HOLDER_INFIX, ".") if _FP32_HOLDER_INFIX in k else k), v)
            for k, v in out
        ]

    def _from_hf_readd_holder(self, hf_state_dict, *args, **kwargs):
        out = _orig_from_hf(self, hf_state_dict, *args, **kwargs)
        renamed = {}
        for k, v in out.items:
            nk = k
            if _FP32_HOLDER_INFIX not in k and ".linear_attn." in k:
                for p in _FP32_HOLDER_PARAMS:
                    suffix = ".linear_attn." + p
                    if k.endswith(suffix):
                        nk = k[: -len(suffix)] + ".linear_attn._fp32_params." + p
                        break
            renamed[nk] = v
        return renamed

    _Adapter.convert_single_tensor_to_hf = _convert_strip_holder
    _Adapter.from_hf = _from_hf_readd_holder
    _FP32_HOLDER_LOAD_PATCH_INSTALLED = True
    print(
        "[tp_plan] patched Qwen3_5MoeStateDictAdapter."
        "{convert_single_tensor_to_hf,from_hf} to LOAD the fp32 SSM holder params "
        "(linear_attn.A_log/dt_bias) from the base checkpoint: strip '_fp32_params' on "
        "the to_hf request key (match disk) and re-add it on from_hf (land the disk "
        "value in the live holder). Fixes zero-valued A_log/dt_bias at refit."
    )


_install_fp32_holder_load_patch


# --- Disk-direct load of the fp32 SSM holder params --------------------------
# The rename above still is not enough. The holder's DTensor lives on a different
# mesh than the DCP planner resolves, so dcp.load never populates it whatever the
# key, and strict=False drops the miss in silence. After the normal load, read the
# two parameters straight from the base safetensors and write them in with
# set_model_state_dict(full_state_dict=True): every rank supplies the full tensor
# and slices its own shard, honouring the parameter's own mesh and bypassing the
# planner. Shares IDEASCIENTIST_FP32_HOLDER_LOAD_FIX.
_FP32_HOLDER_DISK_PATCH_INSTALLED = False


def _install_fp32_holder_disk_load_patch() -> None:
    global _FP32_HOLDER_DISK_PATCH_INSTALLED
    if _FP32_HOLDER_DISK_PATCH_INSTALLED or not _FP32_HOLDER_LOAD_FIX:
        return
    import json as _json

    import torch as _torch
    from nemo_automodel.components.checkpoint.checkpointing import (
        Checkpointer as _Ckpt,
    )
    from nemo_automodel.components.checkpoint.checkpointing import (
        get_safetensors_index_path as _get_idx_path,
    )
    from torch.distributed.checkpoint.state_dict import (
        StateDictOptions as _SDO,
    )
    from torch.distributed.checkpoint.state_dict import (
        set_model_state_dict as _set_sd,
    )

    _orig_load_base = _Ckpt.load_base_model
    _holder_suffixes = tuple(f"._fp32_params.{p}" for p in _FP32_HOLDER_PARAMS)

    def _resolve_ckpt_dir(root_dir, model_name):
        if model_name and os.path.exists(model_name):
            return model_name
        try:
            return _get_idx_path(root_dir, model_name)
        except Exception:
            return None

    def _fill_fp32_holders_from_disk(model, root_dir, model_name):
        ckpt_dir = _resolve_ckpt_dir(root_dir, model_name)
        if not ckpt_dir:
            return
        idx_file = os.path.join(ckpt_dir, "model.safetensors.index.json")
        if not os.path.exists(idx_file):
            return
        with open(idx_file) as _f:
            weight_map = _json.load(_f)["weight_map"]

        # Source FQNs from state_dict -- these are the clean keys set_model_state_dict
        # matches (wrapper prefixes already stripped). The refit-capture from
        # proves the holder param IS enumerated here (it was streamed, at value 0).
        sd = model.state_dict
        live = {k: v for k, v in sd.items if k.endswith(_holder_suffixes)}
        if not live:
            return

        want = {}  # live_key -> disk_key
        for lk in live:
            base = lk.replace(_FP32_HOLDER_INFIX, ".")
            # Candidate disk keys, tried in order until one is in the weight_map.
            # The base checkpoint is a VLM whose language weights live under
            #   model.language_model.layers.*  (verified from the disk index).
            # - VLM trainer live holder strips to model.language_model.layers.* -> matches directly.
            # - TEXT-ONLY trainer live holder strips to model.layers.* (no language_model
            #   segment) -> must be remapped to model.language_model.layers.* to match disk.
            candidates = [base]
            if base.startswith("model.") and ".language_model." not in base:
                candidates.append("model.language_model." + base[len("model.") :])
            candidates.append(
                base[len("model.") :] if base.startswith("model.") else "model." + base
            )
            for dk in candidates:
                if dk in weight_map:
                    want[lk] = dk
                    break
        if not want:
            return

        by_shard: dict[str, list[tuple[str, str]]] = {}
        for lk, dk in want.items:
            by_shard.setdefault(weight_map[dk], []).append((lk, dk))

        from safetensors import safe_open

        full_sd = {}
        for shard, pairs in by_shard.items:
            with safe_open(os.path.join(ckpt_dir, shard), framework="pt") as f:
                for lk, dk in pairs:
                    full_sd[lk] = f.get_tensor(dk).to(live[lk].dtype)

        _set_sd(
            model,
            model_state_dict=full_sd,
            options=_SDO(strict=False, full_state_dict=True),
        )

        # Collective-free sanity log: local-shard norm of one filled param (nonzero on
        # at least one rank => the disk value landed). The authoritative check is the
        # first-refit norm compared against the on-disk value.
        after = model.state_dict
        sample = next(iter(full_sd), None)
        loc = None
        if sample is not None:
            v = after.get(sample)
            try:
                lt = v.to_local if hasattr(v, "to_local") else v
                loc = float(lt.float.norm) if lt is not None else None
            except Exception:
                loc = None
        print(
            f"[tp_plan] fp32 holder disk-load: filled {len(full_sd)} "
            f"holder param(s) from {ckpt_dir} (sample {sample} local-norm={loc})"
        )

    def _patched_load_base_model(
        self, model, device, root_dir, model_name, load_base_model=True
    ):
        _orig_load_base(
            self, model, device, root_dir, model_name, load_base_model=load_base_model
        )
        if not load_base_model or not _FP32_HOLDER_LOAD_FIX:
            return
        try:
            _fill_fp32_holders_from_disk(model, root_dir, model_name)
        except Exception as e:  # never break base load
            print(
                f"[tp_plan] WARNING fp32 holder disk-load failed: {e!r}"
            )

    _Ckpt.load_base_model = _patched_load_base_model
    _FP32_HOLDER_DISK_PATCH_INSTALLED = True
    print(
        "[tp_plan] patched Checkpointer.load_base_model to disk-load the "
        "fp32 SSM holder params (linear_attn._fp32_params.A_log/dt_bias) AFTER base "
        "load -- dcp.load leaves them zero (separate fp32 FSDP group). set_model_state_"
        "dict(full_state_dict=True) distributes the full disk tensor into each local "
        "shard, bypassing the DCP planner. Fixes zero-valued A_log/dt_bias at refit."
    )


_install_fp32_holder_disk_load_patch










def qwen3_5_tp_plan() -> dict:
    """The TP plan for Qwen3.5 dense, as two shards across the mesh.

    1. **lm_head**, vocab-sharded (Colwise ``Shard(-1)``). A replicated lm_head
       forces the logits to a full ``[seq, vocab]`` tensor that is all-gathered
       and fp32-cast at logprob time, which does not fit; sharding keeps them a
       vocab-parallel DTensor so each rank materializes 1/TP of the vocabulary.
       ``use_local_output=False`` keeps the output a DTensor so the memory-
       efficient ``get_logprobs_from_vocab_parallel_logits`` path is taken.

       This is invisible to the colocated-vLLM refit: the send path calls
       ``full_tensor`` on every DTensor before streaming, and the vocabulary
       divides evenly by TP, so vLLM receives the same tensor under the same key
       either way. The shard is a trainer-side placement choice, nothing more.

    2. The per-layer **MLP** (``gate_proj`` / ``up_proj`` Colwise, ``down_proj``
       Rowwise). Sharding only lm_head leaves the transformer replicated on every
       rank, and the backward then has to hold the full
       ``[seq, intermediate_size]`` activation of the MLP recompute, which grows
       with sequence length as trajectories lengthen. The shard divides both that
       activation and the MLP weights by TP without touching sequence length.

    The MLP shard is self-contained — inputs arrive Replicate and outputs return
    Replicate, only the intermediate is sharded — so it composes with the
    replicated residual stream and linear-attention blocks without needing
    sequence-parallel norms. ``translate_to_lora`` auto-wraps Colwise/Rowwise into
    their LoRA variants, sharding base and lora_A/lora_B alike.

    The prefix must match the model's real module paths: torch's
    ``parallelize_module`` raises if a wildcard key matches no module, so a wrong
    prefix fails at startup rather than silently replicating.

    Linear-attention (GatedDeltaNet) projections and full_attention q/k/v/o stay
    replicated — upstream ships no TP plan for them, and the GatedDeltaNet
    backward memory was addressed by installing ``flash-linear-attention`` in the
    trainer venv, which moves the module off its torch fallback onto the fla
    Triton kernels.
    """
    # Module-path prefix for the TEXT-ONLY model (the only mode used by
    # the training overlay). Qwen3_5ForCausalLM nests as
    #   model.model(Qwen3_5TextModel).layers  ->  "model.layers"
    # The VLM class Qwen3_5ForConditionalGeneration nests one level deeper, under
    # "model.language_model.layers"; the trainer does not load the visual tower.
    prefix = "model.layers"
    plan = {
        "lm_head": ColwiseParallel(output_layouts=Shard(-1), use_local_output=False),
        f"{prefix}.*.mlp.gate_proj": ColwiseParallel,
        f"{prefix}.*.mlp.up_proj": ColwiseParallel,
        f"{prefix}.*.mlp.down_proj": RowwiseParallel,
    }

    return plan
