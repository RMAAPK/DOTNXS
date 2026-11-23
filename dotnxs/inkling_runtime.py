"""Experimental NXS-backed CPU components for Inkling 2 Mini.

This module pages real tensors from a remote NXS archive and can execute one
routed MoE layer using the checkpoint's packed NVFP4/MXFP8 experts. It is not a
complete model runtime yet: attention, loop/message state, embeddings, KV and
convolution caches, tokenizer integration, and autoregressive generation are
still required before enabling the chat UI.
"""

from __future__ import annotations

import json
import re
import struct
from typing import Iterable, Iterator

from .remote_archive import DEFAULT_CACHE_BYTES, RemoteNxsArchive

_LAYER_EXPERT_RE = re.compile(r"^model\.llm\.layers\.(\d+)\.mlp\.experts\.")
_PER_EXPERT_TENSORS = (
    "w13_weight",
    "w13_weight_scale",
    "w13_weight_amax",
    "w2_weight",
    "w2_weight_scale",
    "w2_weight_amax",
)
_SHARED_EXPERT_TENSORS = (
    "w13_weight_outer_scale_denominator",
    "w2_weight_outer_scale_denominator",
)


class Inkling2MiniWeights:
    """Metadata and routed-expert fetcher for the Inkling 2 Mini NXS layout."""

    def __init__(self, archive: RemoteNxsArchive) -> None:
        self.archive = archive
        config_bytes = archive.read_entry("asset/config.json")
        try:
            self.config = json.loads(config_bytes.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise ValueError("Inkling config.json in NXS is invalid") from exc
        if not isinstance(self.config, dict) or self.config.get("model_type") != "inkling2_loop":
            raise ValueError("NXS archive is not an Inkling 2 looped-transformer checkpoint")

        text_config = self.config.get("text_config", {})
        self.num_layers = int(text_config.get("num_hidden_layers", 0))
        self.num_experts = int(text_config.get("n_routed_experts", 0))
        self.experts_per_token = int(text_config.get("num_experts_per_tok", 0))
        self.loop_config = text_config.get("loop_config", {})
        if self.num_layers <= 0 or self.num_experts <= 0 or self.experts_per_token <= 0:
            raise ValueError("Inkling config is missing layer/expert routing metadata")

        self.entry_names = {entry["name"] for entry in archive.list_entries()}
        self.expert_layers = sorted(
            {
                int(match.group(1))
                for name in self.entry_names
                if (match := _LAYER_EXPERT_RE.match(name)) is not None
            }
        )
        if not self.expert_layers:
            raise ValueError("NXS archive has no Inkling routed-expert tensor entries")

    @classmethod
    def from_hf_repo(
        cls,
        repo_id: str,
        filename: str = "Inkling-2-Mini.nxs",
        *,
        revision: str = "main",
        cache_bytes: int = DEFAULT_CACHE_BYTES,
        token: str | None = None,
    ) -> "Inkling2MiniWeights":
        archive = RemoteNxsArchive.from_hf_repo(
            repo_id,
            filename,
            revision=revision,
            cache_bytes=cache_bytes,
            token=token,
        )
        try:
            return cls(archive)
        except Exception:
            archive.close()
            raise

    def fetch_expert_tensors(self, layer_id: int, expert_id: int) -> dict[str, bytes]:
        """Fetch one expert's packed weights/scales, verifying NXS chunk hashes."""
        if layer_id not in self.expert_layers:
            raise ValueError(f"layer {layer_id} has no routed-expert tensors in this archive")
        if not 0 <= expert_id < self.num_experts:
            raise ValueError(f"expert_id must be in [0, {self.num_experts})")

        prefix = f"model.llm.layers.{layer_id}.mlp.experts."
        tensors: dict[str, bytes] = {}
        for suffix in _PER_EXPERT_TENSORS:
            name = prefix + suffix
            if name in self.entry_names:
                tensors[suffix] = self.archive.read_tensor_expert(name, expert_id, self.num_experts)
        for suffix in _SHARED_EXPERT_TENSORS:
            name = prefix + suffix
            if name in self.entry_names:
                entry = self.archive.get_entry(name)
                tensors[suffix] = self.archive.read_entry_range(name, 0, entry["nbytes"])

        if "w13_weight" not in tensors or "w2_weight" not in tensors:
            raise ValueError(f"layer {layer_id} does not expose the expected Inkling expert weight tensors")
        return tensors

    def iter_routed_experts(
        self, layer_id: int, expert_ids: Iterable[int]
    ) -> Iterator[tuple[int, dict[str, bytes]]]:
        """Yield selected experts one at a time to bound temporary weight memory."""
        ids = list(expert_ids)
        if not ids:
            return
        if len(ids) > self.experts_per_token:
            raise ValueError(
                f"received {len(ids)} experts, but config routes only {self.experts_per_token} per token"
            )
        if len(set(ids)) != len(ids):
            raise ValueError("expert_ids must be unique")
        for expert_id in ids:
            yield expert_id, self.fetch_expert_tensors(layer_id, expert_id)

    def fetch_routed_experts(self, layer_id: int, expert_ids: Iterable[int]) -> dict[int, dict[str, bytes]]:
        """Fetch a router-selected expert set; use iter_routed_experts to stream one at a time."""
        return dict(self.iter_routed_experts(layer_id, expert_ids))

    def decode_expert_matrices(self, layer_id: int, expert_id: int) -> dict[str, object]:
        """Fetch and decode one expert's stored matrices to CPU float32 arrays.

        The returned ``gate_up`` uses the interleaved row order expected by the
        Transformers Inkling expert layout; ``down`` is the source projection.
        Only one expert is decoded at a time, never the 128-expert layer stack.
        """
        try:
            from .inkling_kernels import (
                dequantize_mxfp8,
                dequantize_nvfp4,
                interleave_gate_up_rows,
            )
        except ImportError as exc:
            raise RuntimeError("Inkling CPU decoding requires NumPy; install dotnxs[runtime]") from exc

        raw = self.fetch_expert_tensors(layer_id, expert_id)
        prefix = f"model.llm.layers.{layer_id}.mlp.experts."
        w13_entry = self.archive.get_entry(prefix + "w13_weight")
        w2_entry = self.archive.get_entry(prefix + "w2_weight")
        w13_shape = tuple(w13_entry["shape"][1:])
        w2_shape = tuple(w2_entry["shape"][1:])

        if w13_entry.get("dtype") == "U8" and w2_entry.get("dtype") == "U8":
            denominator = struct.unpack("<f", raw["w13_weight_outer_scale_denominator"])[0]
            w13 = dequantize_nvfp4(
                raw["w13_weight"],
                raw["w13_weight_scale"],
                struct.unpack("<f", raw["w13_weight_amax"])[0],
                denominator,
                packed_shape=w13_shape,
            )
            w2 = dequantize_nvfp4(
                raw["w2_weight"],
                raw["w2_weight_scale"],
                struct.unpack("<f", raw["w2_weight_amax"])[0],
                struct.unpack("<f", raw["w2_weight_outer_scale_denominator"])[0],
                packed_shape=w2_shape,
            )
        elif w13_entry.get("dtype") == "F8_E4M3" and w2_entry.get("dtype") == "F8_E4M3":
            w13 = dequantize_mxfp8(raw["w13_weight"], raw["w13_weight_scale"], shape=w13_shape)
            w2 = dequantize_mxfp8(raw["w2_weight"], raw["w2_weight_scale"], shape=w2_shape)
        else:
            raise ValueError(
                f"unsupported Inkling expert storage dtypes: {w13_entry.get('dtype')!r}, "
                f"{w2_entry.get('dtype')!r}"
            )

        return {"gate_up": interleave_gate_up_rows(w13), "down": w2}

    def run_expert(self, layer_id: int, expert_id: int, hidden_states):
        """Run one selected expert's CPU SwiGLU path from streamed NXS weights."""
        from .inkling_kernels import run_expert_mlp

        matrices = self.decode_expert_matrices(layer_id, expert_id)
        text_config = self.config["text_config"]
        return run_expert_mlp(
            hidden_states,
            matrices["gate_up"],
            matrices["down"],
            gate_softcap=float(text_config.get("mlp_gate_softcap", 4.0)),
            up_softcap=float(text_config.get("mlp_up_softcap", 25.0)),
        )

    def run_moe_layer(self, layer_id: int, hidden_states):
        """Route one token and execute its selected experts from the NXS archive."""
        try:
            import numpy as np
            from .inkling_kernels import bf16_to_float32, run_sparse_moe
        except ImportError as exc:
            raise RuntimeError("Inkling CPU inference requires NumPy; install dotnxs[runtime]") from exc

        text_config = self.config["text_config"]
        prefix = f"model.llm.layers.{layer_id}.mlp."
        gate_entry = self.archive.get_entry(prefix + "gate.weight")
        gate_shape = tuple(gate_entry.get("shape", ()))
        if len(gate_shape) != 2 or gate_shape[0] != self.num_experts:
            raise ValueError(f"invalid Inkling router weight shape for layer {layer_id}")
        router_weight = bf16_to_float32(self.archive.read_entry(prefix + "gate.weight")).reshape(gate_shape)
        selection_bias = np.frombuffer(
            self.archive.read_entry(prefix + "gate.bias"), dtype="<f4"
        ).copy()

        def load_expert(expert_id: int):
            matrices = self.decode_expert_matrices(layer_id, expert_id)
            return matrices["gate_up"], matrices["down"]

        return run_sparse_moe(
            hidden_states,
            router_weight,
            selection_bias,
            top_k=self.experts_per_token,
            route_scale=float(text_config.get("route_scale", 8.0)),
            expert_loader=load_expert,
            gate_softcap=float(text_config.get("mlp_gate_softcap", 4.0)),
            up_softcap=float(text_config.get("mlp_up_softcap", 25.0)),
        )

    @property
    def cache_stats(self) -> dict[str, int]:
        return self.archive.cache_stats

    def close(self) -> None:
        self.archive.close()

    def __enter__(self) -> "Inkling2MiniWeights":
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> bool:
        self.close()
        return False
