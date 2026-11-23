import importlib.util
import unittest


@unittest.skipUnless(importlib.util.find_spec("numpy"), "install dotnxs[runtime] to test NumPy CPU kernels")
class InklingKernelTests(unittest.TestCase):
    def test_float8_decoders(self):
        from dotnxs.inkling_kernels import decode_e4m3fn, decode_e8m0

        e4m3 = decode_e4m3fn(bytes([0x00, 0x01, 0x38, 0x7E, 0x80, 0xFE, 0x7F]))
        self.assertEqual(e4m3[0], 0.0)
        self.assertAlmostEqual(float(e4m3[1]), 2.0**-9)
        self.assertEqual(e4m3[2], 1.0)
        self.assertEqual(e4m3[3], 448.0)
        self.assertEqual(e4m3[4], 0.0)
        self.assertEqual(e4m3[5], -448.0)
        self.assertTrue(e4m3[6] != e4m3[6])

        e8m0 = decode_e8m0(bytes([126, 127, 128, 255]))
        self.assertEqual(e8m0[:3].tolist(), [0.5, 1.0, 2.0])
        self.assertTrue(e8m0[3] != e8m0[3])

    def test_nvfp4_unpacks_low_then_high_nibble_and_applies_group_and_tensor_scales(self):
        from dotnxs.inkling_kernels import dequantize_nvfp4

        # Each byte contains E2M1 values 0.0 (low nibble) and 0.5 (high nibble).
        result = dequantize_nvfp4(
            bytes([0x10] * 16),
            bytes([0x38, 0x40]),  # E4M3 scales 1.0 and 2.0
            2.0,
            4.0,
            packed_shape=(1, 16),
        )
        expected = [value for _ in range(8) for value in (0.0, 0.25)]
        expected += [value for _ in range(8) for value in (0.0, 0.5)]
        self.assertEqual(result.shape, (1, 32))
        self.assertEqual(result[0].tolist(), expected)

    def test_mxfp8_applies_one_e8m0_scale_per_32_columns(self):
        from dotnxs.inkling_kernels import dequantize_mxfp8

        result = dequantize_mxfp8(
            bytes([0x38] * 64),  # E4M3 1.0
            bytes([127, 128]),  # E8M0 scales 1.0 and 2.0
            shape=(1, 64),
        )
        self.assertEqual(result[0, :32].tolist(), [1.0] * 32)
        self.assertEqual(result[0, 32:].tolist(), [2.0] * 32)

    def test_bf16_reader(self):
        from dotnxs.inkling_kernels import bf16_to_float32

        result = bf16_to_float32(bytes([0x80, 0x3F, 0x00, 0xC0]))
        self.assertEqual(result.tolist(), [1.0, -2.0])

    def test_deinterleaves_gate_up_and_runs_one_swiglu_expert(self):
        import numpy as np

        from dotnxs.inkling_kernels import deinterleave_gate_up_rows, run_expert_mlp

        source_gate_up = np.asarray(
            [[1.0, 0.0], [0.0, 1.0], [2.0, 0.0], [0.0, 2.0]], dtype=np.float32
        )
        gate_up = deinterleave_gate_up_rows(source_gate_up)
        self.assertEqual(
            gate_up.tolist(),
            [[1.0, 0.0], [2.0, 0.0], [0.0, 1.0], [0.0, 2.0]],
        )

        result = run_expert_mlp(np.asarray([1.0, 2.0]), gate_up, np.eye(2, dtype=np.float32))
        gate = 4.0 * np.tanh(np.asarray([1.0, 2.0]) / 4.0)
        up = 25.0 * np.tanh(np.asarray([2.0, 4.0]) / 25.0)
        expected = gate / (1.0 + np.exp(-gate)) * up
        np.testing.assert_allclose(result, expected, rtol=1e-6, atol=1e-6)

    def test_router_uses_selection_bias_only_for_topk_and_ties_by_index(self):
        import numpy as np

        from dotnxs.inkling_kernels import select_experts

        selected, weights = select_experts(
            np.zeros(3, dtype=np.float32),
            np.asarray([0.0, -0.5, 0.25], dtype=np.float32),
            top_k=2,
            route_scale=8.0,
        )
        self.assertEqual(selected.tolist(), [2, 0])
        np.testing.assert_allclose(weights, [4.0, 4.0])

        tied, tied_weights = select_experts(
            np.zeros(3, dtype=np.float32),
            np.zeros(3, dtype=np.float32),
            top_k=2,
            route_scale=8.0,
        )
        self.assertEqual(tied.tolist(), [0, 1])
        np.testing.assert_allclose(tied_weights, [4.0, 4.0])

    def test_rejects_invalid_scale_grid(self):
        from dotnxs.inkling_kernels import dequantize_mxfp8, dequantize_nvfp4

        with self.assertRaisesRegex(ValueError, "divisible by 8"):
            dequantize_nvfp4(bytes(4), bytes(1), 1.0, 1.0, packed_shape=(1, 4))
        with self.assertRaisesRegex(ValueError, "divisible by 32"):
            dequantize_mxfp8(bytes(31), bytes(1), shape=(1, 31))


if __name__ == "__main__":
    unittest.main()
