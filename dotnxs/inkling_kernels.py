"""CPU reference decoders for Inkling 2 Mini's packed MoE expert weights.

The functions here decode one routed expert at a time. They intentionally do
not materialize the full 128-expert stack; callers can release each result
before loading the next expert.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np

# NVFP4 E2M1 values, indexed by the 4-bit code. The low nibble is the even
# column and the high nibble is the odd column in each stored byte.
_E2M1 = np.asarray(
    [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0],
    dtype=np.float32,
)


def _shape_tuple(shape: Sequence[int]) -> tuple[int, ...]:
    result = tuple(shape)
    if not result or any(not isinstance(dim, int) or isinstance(dim, bool) or dim <= 0 for dim in result):
        raise ValueError("tensor shape must contain positive integer dimensions")
    return result


def decode_e4m3fn(data: bytes | bytearray | memoryview | np.ndarray) -> np.ndarray:
    """Decode E4M3-FN bytes to float32, including subnormals and NaN."""
    codes = np.frombuffer(data, dtype=np.uint8) if not isinstance(data, np.ndarray) else np.asarray(data, dtype=np.uint8).reshape(-1)
    sign = np.where((codes & 0x80) != 0, -1.0, 1.0).astype(np.float32)
    exponent = ((codes >> 3) & 0x0F).astype(np.int16)
    mantissa = (codes & 0x07).astype(np.float32)

    values = np.empty(codes.shape, dtype=np.float32)
    subnormal = exponent == 0
    values[subnormal] = mantissa[subnormal] * np.float32(2.0 ** -9)
    normal = ~subnormal
    values[normal] = (1.0 + mantissa[normal] / 8.0) * np.exp2(exponent[normal].astype(np.float32) - 7.0)
    values *= sign
    values[(exponent == 15) & (mantissa == 7)] = np.nan
    return values


def decode_e8m0(data: bytes | bytearray | memoryview | np.ndarray) -> np.ndarray:
    """Decode MXFP8's E8M0 scale bytes to float32 powers of two."""
    codes = np.frombuffer(data, dtype=np.uint8) if not isinstance(data, np.ndarray) else np.asarray(data, dtype=np.uint8).reshape(-1)
    values = np.exp2(codes.astype(np.float32) - 127.0)
    values[codes == 255] = np.nan
    return values.astype(np.float32, copy=False)


def dequantize_nvfp4(
    packed_weights: bytes | bytearray | memoryview | np.ndarray,
    block_scales: bytes | bytearray | memoryview | np.ndarray,
    amax: float,
    outer_scale_denominator: float,
    *,
    packed_shape: Sequence[int],
) -> np.ndarray:
    """Decode one NVFP4 expert matrix to float32.

    ``packed_shape`` is the stored U8 shape ``(out_features, in_features / 2)``.
    Each E4M3 block scale covers 16 unpacked values; the tensor scale is
    ``amax / outer_scale_denominator`` as in the Hugging Face checkpoint
    conversion.
    """
    rows, packed_columns = _shape_tuple(packed_shape)
    if packed_columns % 8:
        raise ValueError("NVFP4 packed input width must be divisible by 8 (16 values per scale block)")
    if not np.isfinite(amax) or not np.isfinite(outer_scale_denominator) or outer_scale_denominator == 0:
        raise ValueError("NVFP4 global scale values must be finite and denominator non-zero")

    packed = np.frombuffer(packed_weights, dtype=np.uint8) if not isinstance(packed_weights, np.ndarray) else np.asarray(packed_weights, dtype=np.uint8).reshape(-1)
    packed = packed.reshape(rows, packed_columns)
    logical_columns = packed_columns * 2
    scales = decode_e4m3fn(block_scales)
    expected_scales = rows * (logical_columns // 16)
    if scales.size != expected_scales:
        raise ValueError(f"NVFP4 scale count mismatch: expected {expected_scales}, got {scales.size}")
    scales = scales.reshape(rows, logical_columns // 16)

    unpacked = np.empty((rows, logical_columns), dtype=np.float32)
    unpacked[:, 0::2] = _E2M1[packed & 0x0F]
    unpacked[:, 1::2] = _E2M1[packed >> 4]
    unpacked = unpacked.reshape(rows, logical_columns // 16, 16)
    unpacked *= scales[..., None]
    unpacked *= np.float32(amax / outer_scale_denominator)
    return unpacked.reshape(rows, logical_columns)


def dequantize_mxfp8(
    weights: bytes | bytearray | memoryview | np.ndarray,
    block_scales: bytes | bytearray | memoryview | np.ndarray,
    *,
    shape: Sequence[int],
) -> np.ndarray:
    """Decode one MXFP8 expert matrix to float32.

    The checkpoint stores E4M3-FN weights and E8M0 scales, one scale for each
    32 adjacent columns.
    """
    rows, columns = _shape_tuple(shape)
    if columns % 32:
        raise ValueError("MXFP8 input width must be divisible by 32")
    values = decode_e4m3fn(weights)
    if values.size != rows * columns:
        raise ValueError(f"MXFP8 weight count mismatch: expected {rows * columns}, got {values.size}")
    scales = decode_e8m0(block_scales)
    expected_scales = rows * (columns // 32)
    if scales.size != expected_scales:
        raise ValueError(f"MXFP8 scale count mismatch: expected {expected_scales}, got {scales.size}")
    values = values.reshape(rows, columns // 32, 32)
    values *= scales.reshape(rows, columns // 32, 1)
    return values.reshape(rows, columns)


def bf16_to_float32(data: bytes | bytearray | memoryview | np.ndarray) -> np.ndarray:
    """Interpret little-endian BF16 bytes as float32 without rounding."""
    values = np.frombuffer(data, dtype="<u2") if not isinstance(data, np.ndarray) else np.asarray(data, dtype="<u2").reshape(-1)
    expanded = values.astype(np.uint32) << 16
    return expanded.view(np.float32)


def interleave_gate_up_rows(weight: np.ndarray) -> np.ndarray:
    """Convert checkpoint's contiguous gate/up rows to the HF interleaved layout."""
    matrix = np.asarray(weight)
    if matrix.ndim != 2 or matrix.shape[0] % 2:
        raise ValueError("gate/up projection must be a matrix with an even row count")
    half = matrix.shape[0] // 2
    return matrix.reshape(2, half, matrix.shape[1]).transpose(1, 0, 2).reshape(matrix.shape)


def select_experts(
    router_logits: np.ndarray,
    selection_bias: np.ndarray,
    *,
    top_k: int,
    route_scale: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Select by sigmoid score plus selection bias; normalize unbiased scores."""
    logits = np.asarray(router_logits, dtype=np.float32).reshape(-1)
    bias = np.asarray(selection_bias, dtype=np.float32).reshape(-1)
    if logits.size == 0 or logits.shape != bias.shape:
        raise ValueError("router logits and selection bias must have the same non-empty shape")
    if not 1 <= top_k <= logits.size:
        raise ValueError("top_k must be between one and the number of experts")
    if not np.isfinite(route_scale) or route_scale <= 0:
        raise ValueError("route_scale must be finite and positive")

    probabilities = np.empty_like(logits)
    positive = logits >= 0
    probabilities[positive] = 1.0 / (1.0 + np.exp(-logits[positive]))
    exp_logits = np.exp(logits[~positive])
    probabilities[~positive] = exp_logits / (1.0 + exp_logits)
    scores = probabilities + bias
    expert_ids = np.arange(logits.size)
    # lexsort's last key is primary; use the ID as the deterministic tie-breaker.
    selected = np.lexsort((expert_ids, -scores))[:top_k]
    weights = probabilities[selected] / (probabilities[selected].sum() + 1e-20) * route_scale
    return selected.astype(np.int64), weights.astype(np.float32)


def run_expert_mlp(
    hidden_states: np.ndarray,
    gate_up_weight: np.ndarray,
    down_weight: np.ndarray,
    *,
    gate_softcap: float = 4.0,
    up_softcap: float = 25.0,
) -> np.ndarray:
    """Run one Inkling SwiGLU expert using decoded float32 matrices.

    ``gate_up_weight`` is in the HF interleaved row layout: gate and up rows
    alternate. The input and output are single-token vectors; this deliberately
    keeps the reference path small enough to consume one expert at a time.
    """
    x = np.asarray(hidden_states, dtype=np.float32).reshape(-1)
    gate_up = np.asarray(gate_up_weight, dtype=np.float32)
    down = np.asarray(down_weight, dtype=np.float32)
    if gate_up.ndim != 2 or down.ndim != 2:
        raise ValueError("expert projection weights must be rank-2 matrices")
    if gate_up.shape[0] % 2 or gate_up.shape[1] != x.size:
        raise ValueError("gate/up matrix dimensions do not match the hidden state")
    intermediate = gate_up.shape[0] // 2
    if down.shape != (x.size, intermediate):
        raise ValueError("down projection dimensions do not match the hidden state and intermediate width")
    if gate_softcap <= 0 or up_softcap <= 0:
        raise ValueError("expert softcap values must be positive")

    projections = gate_up @ x
    gate = projections[0::2]
    up = projections[1::2]
    gate = gate_softcap * np.tanh(gate / gate_softcap)
    up = up_softcap * np.tanh(up / up_softcap)
    # Numerically stable SiLU: g * sigmoid(g).
    sigmoid_gate = np.empty_like(gate)
    positive = gate >= 0
    sigmoid_gate[positive] = 1.0 / (1.0 + np.exp(-gate[positive]))
    exp_gate = np.exp(gate[~positive])
    sigmoid_gate[~positive] = exp_gate / (1.0 + exp_gate)
    return down @ (gate * sigmoid_gate * up)


def run_sparse_moe(
    hidden_states: np.ndarray,
    router_weight: np.ndarray,
    selection_bias: np.ndarray,
    *,
    top_k: int,
    route_scale: float,
    expert_loader,
    gate_softcap: float = 4.0,
    up_softcap: float = 25.0,
) -> np.ndarray:
    """Run one token through top-k experts, loading and releasing each expert in turn."""
    x = np.asarray(hidden_states, dtype=np.float32).reshape(-1)
    router = np.asarray(router_weight, dtype=np.float32)
    if router.ndim != 2 or router.shape[1] != x.size:
        raise ValueError("router weight dimensions do not match the hidden state")
    expert_ids, route_weights = select_experts(
        router @ x,
        selection_bias,
        top_k=top_k,
        route_scale=route_scale,
    )
    output = np.zeros(x.shape, dtype=np.float32)
    for expert_id, route_weight in zip(expert_ids, route_weights):
        gate_up_weight, down_weight = expert_loader(int(expert_id))
        output += route_weight * run_expert_mlp(
            x,
            gate_up_weight,
            down_weight,
            gate_softcap=gate_softcap,
            up_softcap=up_softcap,
        )
    return output


__all__ = [
    "bf16_to_float32",
    "decode_e4m3fn",
    "decode_e8m0",
    "dequantize_mxfp8",
    "dequantize_nvfp4",
    "interleave_gate_up_rows",
    "run_expert_mlp",
    "run_sparse_moe",
    "select_experts",
]
