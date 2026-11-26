"""CPU reference decoders for Inkling 2 Mini's packed MoE expert weights.

The functions here decode one routed expert at a time. They intentionally do
not materialize the full 128-expert stack; callers can release each result
before loading the next expert.
"""

from __future__ import annotations

import math
import struct
from typing import Sequence

import numpy as np

try:
    from numba import njit as _njit
except ImportError:  # The ordinary NumPy reference path remains available without Numba.
    _njit = None

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


def rms_norm(hidden_states: np.ndarray, weight: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    """Apply the weighted RMSNorm used by Inkling's embedding and output head."""
    values = np.asarray(hidden_states, dtype=np.float32)
    scale = np.asarray(weight, dtype=np.float32)
    if values.shape != scale.shape or values.ndim != 1:
        raise ValueError("RMSNorm expects matching one-dimensional hidden and weight vectors")
    if not np.isfinite(eps) or eps <= 0:
        raise ValueError("RMSNorm epsilon must be finite and positive")
    return values * np.float32(1.0 / np.sqrt(np.mean(values * values) + eps)) * scale


def deinterleave_gate_up_rows(weight: np.ndarray) -> np.ndarray:
    """Convert checkpoint's alternating gate/up rows to contiguous HF gate/up rows."""
    matrix = np.asarray(weight)
    if matrix.ndim != 2 or matrix.shape[0] % 2:
        raise ValueError("gate/up projection must be a matrix with an even row count")
    half = matrix.shape[0] // 2
    return matrix.reshape(half, 2, matrix.shape[1]).transpose(1, 0, 2).reshape(matrix.shape)


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


def round_bf16(values: np.ndarray) -> np.ndarray:
    """Round float32 values to BF16 precision and return them in float32 storage."""
    values = np.ascontiguousarray(values, dtype=np.float32)
    bits = values.view(np.uint32)
    rounded = bits + np.uint32(0x7FFF) + ((bits >> 16) & 1)
    return (rounded & np.uint32(0xFFFF0000)).view(np.float32)


def run_expert_mlp(
    hidden_states: np.ndarray,
    gate_up_weight: np.ndarray,
    down_weight: np.ndarray,
    *,
    gate_softcap: float = 4.0,
    up_softcap: float = 25.0,
    bfloat16: bool = False,
) -> np.ndarray:
    """Run one Inkling SwiGLU expert using decoded float32 matrices.

    ``gate_up_weight`` is in the HF layout: all gate rows followed by all up
    rows. The input and output are single-token vectors; this deliberately keeps
    the reference path small enough to consume one expert at a time.
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

    if bfloat16:
        x = round_bf16(x)
        gate_up = round_bf16(gate_up)
        down = round_bf16(down)
    projections = gate_up @ x
    if bfloat16:
        projections = round_bf16(projections)
    gate = projections[:intermediate]
    up = projections[intermediate:]
    gate = gate_softcap * np.tanh(gate / gate_softcap)
    up = up_softcap * np.tanh(up / up_softcap)
    # Numerically stable SiLU: g * sigmoid(g).
    sigmoid_gate = np.empty_like(gate)
    positive = gate >= 0
    sigmoid_gate[positive] = 1.0 / (1.0 + np.exp(-gate[positive]))
    exp_gate = np.exp(gate[~positive])
    sigmoid_gate[~positive] = exp_gate / (1.0 + exp_gate)
    activations = gate * sigmoid_gate * up
    if bfloat16:
        activations = round_bf16(activations)
    output = down @ activations
    return round_bf16(output) if bfloat16 else output


def run_selected_experts(
    hidden_states: np.ndarray,
    selected_experts: np.ndarray,
    routing_weights: np.ndarray,
    *,
    expert_loader,
    gate_softcap: float = 4.0,
    up_softcap: float = 25.0,
    bfloat16: bool = True,
) -> np.ndarray:
    """Execute routed experts for a token batch, materializing one expert at a time."""
    x = np.asarray(hidden_states, dtype=np.float32)
    ids = np.asarray(selected_experts, dtype=np.int64)
    weights = np.asarray(routing_weights, dtype=np.float32)
    if x.ndim != 2 or ids.ndim != 2 or ids.shape != weights.shape or ids.shape[0] != x.shape[0]:
        raise ValueError("hidden states, selected experts, and routing weights have incompatible shapes")
    output = np.zeros_like(x, dtype=np.float32)
    for expert_id in np.unique(ids):
        gate_up_weight, down_weight = expert_loader(int(expert_id))
        token_indices, slots = np.where(ids == expert_id)
        for token_index, slot in zip(token_indices, slots):
            output[token_index] += weights[token_index, slot] * run_expert_mlp(
                x[token_index],
                gate_up_weight,
                down_weight,
                gate_softcap=gate_softcap,
                up_softcap=up_softcap,
                bfloat16=bfloat16,
            )
    return output


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
    return run_selected_experts(
        x[None, :],
        expert_ids[None, :],
        route_weights[None, :],
        expert_loader=expert_loader,
        gate_softcap=gate_softcap,
        up_softcap=up_softcap,
        bfloat16=False,
    )[0]


if _njit is not None:
    @_njit(cache=True)
    def _numba_round_bf16_scalar(value):
        value = np.float32(value)
        if value == 0.0 or not math.isfinite(value):
            return value
        if abs(value) < np.float32(2.0 ** -126):
            quantum = np.float32(2.0 ** -133)
            return np.float32(round(value / quantum) * quantum)
        mantissa, exponent = math.frexp(value)
        rounded_mantissa = np.float32(round(mantissa * 256.0) / 256.0)
        return np.float32(math.ldexp(rounded_mantissa, exponent))

    @_njit(cache=True)
    def _numba_e4m3(code):
        code = int(code)
        sign = -1.0 if code & 0x80 else 1.0
        exponent = (code >> 3) & 0x0F
        mantissa = code & 0x07
        if exponent == 15 and mantissa == 7:
            return np.float32(np.nan)
        if exponent == 0:
            value = mantissa * (2.0 ** -9)
        else:
            value = (1.0 + mantissa / 8.0) * (2.0 ** (exponent - 7))
        return np.float32(sign * value)

    @_njit(cache=True)
    def _numba_e8m0(code):
        code = int(code)
        if code == 255:
            return np.float32(np.nan)
        return np.float32(math.ldexp(1.0, code - 127))

    @_njit(cache=True)
    def _numba_nvfp4_matvec(weights, scales, values, amax, denominator, bf16):
        rows, packed_columns = weights.shape
        output = np.empty(rows, dtype=np.float32)
        tensor_scale = np.float32(amax / denominator)
        e2m1 = (0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0)
        for row in range(rows):
            total = np.float32(0.0)
            for column in range(packed_columns):
                byte = int(weights[row, column])
                nibble_low = byte & 0x0F
                nibble_high = byte >> 4
                scale = np.float32(_numba_e4m3(scales[row, column // 8]) * tensor_scale)
                weight_low = np.float32(e2m1[nibble_low & 7])
                weight_high = np.float32(e2m1[nibble_high & 7])
                if nibble_low & 8:
                    weight_low = -weight_low
                if nibble_high & 8:
                    weight_high = -weight_high
                weight_low = np.float32(weight_low * scale)
                weight_high = np.float32(weight_high * scale)
                if bf16:
                    weight_low = _numba_round_bf16_scalar(weight_low)
                    weight_high = _numba_round_bf16_scalar(weight_high)
                total = np.float32(total + weight_low * values[column * 2])
                total = np.float32(total + weight_high * values[column * 2 + 1])
            output[row] = _numba_round_bf16_scalar(total) if bf16 else total
        return output

    @_njit(cache=True)
    def _numba_mxfp8_matvec(weights, scales, values, bf16):
        rows, columns = weights.shape
        output = np.empty(rows, dtype=np.float32)
        for row in range(rows):
            total = np.float32(0.0)
            for column in range(columns):
                weight = np.float32(_numba_e4m3(weights[row, column]))
                scale = _numba_e8m0(scales[row, column // 32])
                weight = np.float32(weight * scale)
                if bf16:
                    weight = _numba_round_bf16_scalar(weight)
                total = np.float32(total + weight * values[column])
            output[row] = _numba_round_bf16_scalar(total) if bf16 else total
        return output
else:
    _numba_round_bf16_scalar = None
    _numba_e4m3 = None
    _numba_e8m0 = None
    _numba_nvfp4_matvec = None
    _numba_mxfp8_matvec = None


def run_file_backed_expert_mlp(
    hidden_states,
    tensors: dict[str, object],
    *,
    w13_shape: Sequence[int],
    w2_shape: Sequence[int],
    storage_dtype: str,
    gate_softcap: float = 4.0,
    up_softcap: float = 25.0,
    bfloat16: bool = True,
) -> np.ndarray:
    """Run one expert directly from memory-mapped packed weight/scales files.

    Numba decodes packed weights inside each matvec, avoiding full float32
    copies of the expert matrices. If Numba is unavailable, use the existing
    NumPy reference decoder as a correctness fallback.
    """
    x = np.asarray(hidden_states, dtype=np.float32).reshape(-1)
    w13_rows, w13_columns = _shape_tuple(w13_shape)
    w2_rows, w2_columns = _shape_tuple(w2_shape)
    if w13_rows % 2 or w2_rows != x.size:
        raise ValueError("file-backed expert matrix shapes do not match hidden/intermediate widths")
    if gate_softcap <= 0 or up_softcap <= 0:
        raise ValueError("expert softcap values must be positive")
    w13 = np.asarray(tensors["w13_weight"], dtype=np.uint8).reshape(w13_rows, w13_columns)
    w2 = np.asarray(tensors["w2_weight"], dtype=np.uint8).reshape(w2_rows, w2_columns)
    w13_scales = np.asarray(tensors["w13_weight_scale"], dtype=np.uint8).reshape(w13_rows, -1)
    w2_scales = np.asarray(tensors["w2_weight_scale"], dtype=np.uint8).reshape(w2_rows, -1)

    if storage_dtype == "U8":
        w13_amax = struct.unpack("<f", tensors["w13_weight_amax"])[0]
        w2_amax = struct.unpack("<f", tensors["w2_weight_amax"])[0]
        w13_denominator = struct.unpack("<f", tensors["w13_weight_outer_scale_denominator"])[0]
        w2_denominator = struct.unpack("<f", tensors["w2_weight_outer_scale_denominator"])[0]
        expected_w13_scales = w13_rows * ((w13_columns * 2) // 16)
        expected_w2_scales = w2_rows * ((w2_columns * 2) // 16)
        if (
            x.size != w13_columns * 2
            or w13_rows % 4
            or w13_rows // 4 != w2_columns
            or w13_columns % 8
            or w2_columns % 8
            or w13_scales.size != expected_w13_scales
            or w2_scales.size != expected_w2_scales
        ):
            raise ValueError("NVFP4 mapped weights or scales have invalid shapes")
        if (
            not np.isfinite(w13_amax)
            or not np.isfinite(w2_amax)
            or not np.isfinite(w13_denominator)
            or not np.isfinite(w2_denominator)
            or w13_denominator == 0
            or w2_denominator == 0
        ):
            raise ValueError("NVFP4 tensor scales must be finite with non-zero denominators")
        if _numba_nvfp4_matvec is None:
            w13_float = dequantize_nvfp4(
                w13, w13_scales, w13_amax, w13_denominator, packed_shape=(w13_rows, w13_columns)
            )
            w2_float = dequantize_nvfp4(
                w2, w2_scales, w2_amax, w2_denominator, packed_shape=(w2_rows, w2_columns)
            )
        else:
            if bfloat16:
                x = round_bf16(x)
            raw_projection = _numba_nvfp4_matvec(
                w13, w13_scales, x, w13_amax, w13_denominator, bfloat16
            )
            gate_up = np.concatenate((raw_projection[0::2], raw_projection[1::2]))
            intermediate = gate_up.size // 2
            gate, up = gate_up[:intermediate], gate_up[intermediate:]
            gate = np.float32(gate_softcap) * np.tanh(gate / np.float32(gate_softcap))
            up = np.float32(up_softcap) * np.tanh(up / np.float32(up_softcap))
            sigmoid = np.empty_like(gate)
            positive = gate >= 0
            sigmoid[positive] = 1.0 / (1.0 + np.exp(-gate[positive]))
            exp_gate = np.exp(gate[~positive])
            sigmoid[~positive] = exp_gate / (1.0 + exp_gate)
            activations = gate * sigmoid * up
            if bfloat16:
                activations = round_bf16(activations)
            return _numba_nvfp4_matvec(
                w2, w2_scales, activations, w2_amax, w2_denominator, bfloat16
            )
    elif storage_dtype == "F8_E4M3":
        expected_w13_scales = w13_rows * (w13_columns // 32)
        expected_w2_scales = w2_rows * (w2_columns // 32)
        if (
            x.size != w13_columns
            or w13_rows // 2 != w2_columns
            or w13_columns % 32
            or w2_columns % 32
            or w13_scales.size != expected_w13_scales
            or w2_scales.size != expected_w2_scales
        ):
            raise ValueError("MXFP8 mapped weights or scales have invalid shapes")
        if _numba_mxfp8_matvec is None:
            w13_float = dequantize_mxfp8(w13, w13_scales, shape=(w13_rows, w13_columns))
            w2_float = dequantize_mxfp8(w2, w2_scales, shape=(w2_rows, w2_columns))
        else:
            if bfloat16:
                x = round_bf16(x)
            raw_projection = _numba_mxfp8_matvec(w13, w13_scales, x, bfloat16)
            gate_up = np.concatenate((raw_projection[0::2], raw_projection[1::2]))
            intermediate = gate_up.size // 2
            gate, up = gate_up[:intermediate], gate_up[intermediate:]
            gate = np.float32(gate_softcap) * np.tanh(gate / np.float32(gate_softcap))
            up = np.float32(up_softcap) * np.tanh(up / np.float32(up_softcap))
            sigmoid = np.empty_like(gate)
            positive = gate >= 0
            sigmoid[positive] = 1.0 / (1.0 + np.exp(-gate[positive]))
            exp_gate = np.exp(gate[~positive])
            sigmoid[~positive] = exp_gate / (1.0 + exp_gate)
            activations = gate * sigmoid * up
            if bfloat16:
                activations = round_bf16(activations)
            return _numba_mxfp8_matvec(w2, w2_scales, activations, bfloat16)
    else:
        raise ValueError(f"unsupported file-backed expert dtype {storage_dtype!r}")

    gate_up = deinterleave_gate_up_rows(w13_float)
    return run_expert_mlp(
        x,
        gate_up,
        w2_float,
        gate_softcap=gate_softcap,
        up_softcap=up_softcap,
        bfloat16=bfloat16,
    )


__all__ = [
    "bf16_to_float32",
    "decode_e4m3fn",
    "decode_e8m0",
    "dequantize_mxfp8",
    "dequantize_nvfp4",
    "deinterleave_gate_up_rows",
    "run_expert_mlp",
    "run_file_backed_expert_mlp",
    "run_selected_experts",
    "run_sparse_moe",
    "round_bf16",
    "rms_norm",
    "select_experts",
]
