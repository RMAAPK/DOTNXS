"""Experimental full-text Inkling runner with NXS-backed, per-layer CPU loading.

The official Transformers implementation supplies the attention, loop, cache,
and message-passing math. This adapter keeps dense layer tensors ephemeral,
streams each selected MoE expert from the NXS archive, and reads the vocabulary
head in bounded row blocks. It is intentionally CPU-only and serializes requests
because layer parameters are swapped in and out during each forward pass.
"""

from __future__ import annotations

import re
import tempfile
import threading
import types
from pathlib import Path

from .inkling_runtime import Inkling2MiniWeights

_LAYER_SOURCE_RE = re.compile(r"^model\.llm\.layers\.(\d+)\.")


class StreamedInklingRunner:
    """Run text-only Inkling 2 Mini forwards using weights paged from one NXS file."""

    def __init__(
        self,
        weights: Inkling2MiniWeights,
        *,
        max_context_tokens: int = 128,
    ) -> None:
        if max_context_tokens <= 0:
            raise ValueError("max_context_tokens must be positive")
        self.weights = weights
        self.max_context_tokens = max_context_tokens
        self._lock = threading.RLock()
        self._tokenizer_tempdir = None
        self._build_transformer()
        self._load_global_weights()
        self._install_streaming_experts()
        self._install_streaming_io()
        self._install_layer_hooks()
        self.tokenizer = self._load_tokenizer()

    @classmethod
    def from_hf_repo(
        cls,
        repo_id: str = "rmaaorg/Inkling-2-Mini-NXS",
        filename: str = "Inkling-2-Mini.nxs",
        *,
        revision: str = "main",
        cache_bytes: int = 200 * 1024 * 1024,
        max_context_tokens: int = 128,
        token: str | None = None,
    ) -> "StreamedInklingRunner":
        weights = Inkling2MiniWeights.from_hf_repo(
            repo_id,
            filename,
            revision=revision,
            cache_bytes=cache_bytes,
            token=token,
        )
        try:
            return cls(weights, max_context_tokens=max_context_tokens)
        except Exception:
            weights.close()
            raise

    def _build_transformer(self) -> None:
        try:
            import torch
            from transformers.conversion_mapping import get_checkpoint_conversion_mapping
            from transformers.core_model_loading import WeightConverter, WeightRenaming, rename_source_key
            from transformers.models.inkling2_loop.configuration_inkling2_loop import Inkling2LoopTextConfig
            from transformers.models.inkling2_loop.modeling_inkling2_loop import Inkling2LoopForCausalLM
        except ImportError as exc:
            raise RuntimeError(
                "Streamed Inkling requires PyTorch and Transformers 5.24+ with Inkling2 support"
            ) from exc

        self.torch = torch
        text_config = dict(self.weights.config["text_config"])
        text_config.pop("model_type", None)
        config = Inkling2LoopTextConfig(**text_config)

        previous_dtype = torch.get_default_dtype()
        torch.set_default_dtype(torch.bfloat16)
        try:
            with torch.device("meta"):
                self.model = Inkling2LoopForCausalLM(config)
        finally:
            torch.set_default_dtype(previous_dtype)
        self.model.eval()
        self._meta_state = self.model.state_dict()
        self._templates = {
            name: tensor for name, tensor in self.model.named_parameters()
        }
        self._templates.update({name: tensor for name, tensor in self.model.named_buffers()})

        conversions = get_checkpoint_conversion_mapping("inkling2_loop_text") or []
        self._renamings = [item for item in conversions if isinstance(item, WeightRenaming)]
        self._converters = [item for item in conversions if isinstance(item, WeightConverter)]
        self._rename_source_key = rename_source_key
        self._layer_entries: dict[int, list[tuple[str, str]]] = {}
        self._global_entries: list[tuple[str, str]] = []
        self._message_query_entries: dict[str, dict] = {}
        self._index_entries()

    def _map_source(self, source_name: str) -> tuple[str, str | None]:
        return self._rename_source_key(
            source_name,
            self._renamings,
            self._converters,
            base_model_prefix=self.model.base_model_prefix,
            meta_state_dict=self._meta_state,
        )

    def _index_entries(self) -> None:
        layer_entries: dict[int, list[tuple[str, str]]] = {}
        global_entries: list[tuple[str, str]] = []
        message_sources = {
            "model.llm.middle_cycle_cross_pass_queries",
            "model.llm.middle_cycle_cross_pass_kv_queries",
        }
        for entry in self.weights.archive.list_entries(kind="tensor"):
            source = entry["name"]
            if source in message_sources:
                self._message_query_entries[source] = entry
                continue

            target, conversion = self._map_source(source)
            # Experts are loaded only after routing. The large embedding and head
            # are also replaced with row-wise archive readers below.
            if conversion is not None:
                continue
            if source in {"model.llm.embed.weight", "model.llm.embed_norm.weight", "model.llm.unembed.weight"}:
                continue
            if target not in self._templates:
                continue

            match = _LAYER_SOURCE_RE.match(source)
            if match:
                layer_entries.setdefault(int(match.group(1)), []).append((source, target))
            else:
                global_entries.append((source, target))
        self._layer_entries = layer_entries
        self._global_entries = global_entries

    def _tensor_from_entry(self, source_name: str, target_name: str):
        entry = self.weights.archive.get_entry(source_name)
        source_dtypes = {
            "BF16": self.torch.bfloat16,
            "F32": self.torch.float32,
            "F16": self.torch.float16,
            "U8": self.torch.uint8,
        }
        dtype = source_dtypes.get(entry.get("dtype"))
        if dtype is None:
            raise ValueError(f"unsupported non-expert dtype {entry.get('dtype')!r} for {source_name}")
        shape = entry.get("shape")
        if not isinstance(shape, list):
            raise ValueError(f"tensor {source_name} has no shape metadata")
        payload = self.weights.archive.read_entry(source_name, max_bytes=64 * 1024 * 1024)
        tensor = self.torch.frombuffer(bytearray(payload), dtype=dtype).reshape(shape)
        template = self._templates[target_name]
        if tuple(tensor.shape) != tuple(template.shape):
            raise ValueError(
                f"checkpoint shape mismatch for {source_name} -> {target_name}: "
                f"{tuple(tensor.shape)} != {tuple(template.shape)}"
            )
        if tensor.dtype != template.dtype:
            tensor = tensor.to(dtype=template.dtype)
        return tensor

    def _replace_tensor(self, target_name: str, tensor, stash: list | None = None) -> None:
        module_name, leaf = target_name.rsplit(".", 1)
        module = self.model.get_submodule(module_name)
        if leaf in module._parameters:
            previous = module._parameters[leaf]
            if stash is not None:
                stash.append((module, leaf, "parameter", previous))
            module._parameters[leaf] = self.torch.nn.Parameter(tensor, requires_grad=False)
        elif leaf in module._buffers:
            previous = module._buffers[leaf]
            if stash is not None:
                stash.append((module, leaf, "buffer", previous))
            module._buffers[leaf] = tensor
        else:
            raise KeyError(f"mapped target is not a model parameter or buffer: {target_name}")

    @staticmethod
    def _restore_tensors(stash: list) -> None:
        for module, leaf, kind, previous in reversed(stash):
            if kind == "parameter":
                module._parameters[leaf] = previous
            else:
                module._buffers[leaf] = previous

    def _load_global_weights(self) -> None:
        for source, target in self._global_entries:
            tensor = self._tensor_from_entry(source, target)
            self._replace_tensor(target, tensor)

        if self._message_query_entries:
            required = {
                "model.llm.middle_cycle_cross_pass_queries",
                "model.llm.middle_cycle_cross_pass_kv_queries",
            }
            if set(self._message_query_entries) != required:
                raise ValueError("NXS archive must contain both loop message-query tensors")
            query = self._read_float32(self._message_query_entries["model.llm.middle_cycle_cross_pass_queries"])
            key_value = self._read_float32(
                self._message_query_entries["model.llm.middle_cycle_cross_pass_kv_queries"]
            )
            hidden_size = int(self.weights.config["text_config"]["hidden_size"])
            query = query.reshape(-1, 2, hidden_size)
            key_value = key_value.reshape(-1, 2, hidden_size)
            merged = self.torch.cat((query[:, :1], key_value, query[:, 1:]), dim=1)
            self._replace_tensor("model.message_input.query", merged)

    def _read_float32(self, entry: dict):
        import numpy as np

        data = self.weights.archive.read_entry(entry["name"], max_bytes=64 * 1024 * 1024)
        values = np.frombuffer(data, dtype="<f4").copy()
        return self.torch.from_numpy(values).reshape(entry["shape"])

    def _install_streaming_experts(self) -> None:
        import numpy as np

        for layer_idx, layer in enumerate(self.model.model.layers):
            expert_module = layer.mlp.experts

            def streamed_forward(_module, hidden_states, top_k_index, top_k_weights, idx=layer_idx):
                return self._run_routed_experts(idx, hidden_states, top_k_index, top_k_weights, np)

            expert_module.forward = types.MethodType(streamed_forward, expert_module)

    def _run_routed_experts(self, layer_idx: int, hidden_states, top_k_index, top_k_weights, np):
        hidden = hidden_states.detach().float().cpu().numpy()
        selected = top_k_index.detach().cpu().numpy()
        weights = top_k_weights.detach().float().cpu().numpy()
        result = self.weights.run_routed_experts(
            layer_idx,
            hidden,
            selected,
            weights,
            bfloat16=True,
        )
        return self.torch.from_numpy(result).to(device=hidden_states.device, dtype=hidden_states.dtype)

    def _install_streaming_io(self) -> None:
        torch = self.torch
        weights = self.weights

        class StreamingEmbedding(torch.nn.Module):
            def forward(self, input_ids):
                import numpy as np

                ids = input_ids.detach().cpu().reshape(-1).tolist()
                rows = np.stack([weights.embed_token(int(token_id)) for token_id in ids])
                values = torch.from_numpy(rows).to(dtype=torch.bfloat16)
                return values.reshape(*input_ids.shape, rows.shape[-1])

        class StreamingLMHead(torch.nn.Module):
            def forward(self, hidden_states):
                import numpy as np

                shape = hidden_states.shape[:-1]
                flat = hidden_states.detach().float().cpu().reshape(-1, hidden_states.shape[-1]).numpy()
                logits = [weights.project_logits(row, apply_final_norm=False) for row in flat]
                values = torch.from_numpy(np.stack(logits))
                return values.reshape(*shape, values.shape[-1])

        self.model.model.embed_tokens = StreamingEmbedding()
        self.model.lm_head = StreamingLMHead()
        # project_logits includes the model's final logit softcap.
        self.model.logit_softcap = torch.nn.Identity()

    def _install_layer_hooks(self) -> None:
        for layer_idx, layer in enumerate(self.model.model.layers):
            original_forward = layer.forward

            def streamed_forward(*args, _idx=layer_idx, _original=original_forward, **kwargs):
                with self._lock:
                    stash = self._load_layer(_idx)
                    try:
                        return _original(*args, **kwargs)
                    finally:
                        self._restore_tensors(stash)

            layer.forward = streamed_forward

    def _load_layer(self, layer_idx: int) -> list:
        stash = []
        try:
            layer_entries = self._layer_entries.get(layer_idx, ())
            self.weights.prefetch_entries(source for source, _target in layer_entries)
            for source, target in layer_entries:
                tensor = self._tensor_from_entry(source, target)
                self._replace_tensor(target, tensor, stash)
            return stash
        except Exception:
            self._restore_tensors(stash)
            raise

    def _load_tokenizer(self):
        try:
            from transformers import AutoTokenizer
        except ImportError as exc:
            raise RuntimeError("Text generation requires the Transformers tokenizer package") from exc

        self._tokenizer_tempdir = tempfile.TemporaryDirectory(prefix="dotnxs-tokenizer-")
        folder = Path(self._tokenizer_tempdir.name)
        copied = False
        for filename in (
            "tokenizer.json",
            "tokenizer_config.json",
            "special_tokens_map.json",
            "chat_template.jinja",
            "generation_config.json",
        ):
            entry_name = f"asset/{filename}"
            if entry_name in self.weights.entry_names:
                (folder / filename).write_bytes(self.weights.archive.read_entry(entry_name))
                copied = True
        if not copied or not (folder / "tokenizer.json").exists():
            raise ValueError("NXS archive does not include the tokenizer assets required for text chat")
        tokenizer = AutoTokenizer.from_pretrained(
            folder,
            use_fast=True,
            local_files_only=True,
            trust_remote_code=False,
        )
        template_path = folder / "chat_template.jinja"
        if template_path.exists():
            tokenizer.chat_template = template_path.read_text(encoding="utf-8")
        return tokenizer

    def stream_chat(self, messages: list[dict[str, str]], *, max_new_tokens: int = 24):
        """Yield the growing assistant reply; uses greedy decoding and a bounded context."""
        if not isinstance(max_new_tokens, int) or not 1 <= max_new_tokens <= 128:
            raise ValueError("max_new_tokens must be between 1 and 128")
        with self._lock, self.torch.inference_mode():
            encoded = self.tokenizer.apply_chat_template(
                messages,
                tokenize=True,
                add_generation_prompt=True,
                return_tensors="pt",
                reasoning_effort="low",
            )
            try:
                input_ids = encoded["input_ids"]
            except (KeyError, IndexError, TypeError):
                input_ids = encoded
            if input_ids.ndim == 1:
                input_ids = input_ids.unsqueeze(0)
            if input_ids.shape[-1] > self.max_context_tokens:
                raise ValueError(
                    f"Prompt has {input_ids.shape[-1]} tokens; this demo is limited to "
                    f"{self.max_context_tokens} tokens per request."
                )

            cache = None
            next_input = input_ids.to(dtype=self.torch.long, device="cpu")
            generated: list[int] = []
            eos_id = self.weights.config.get("eos_token_id")
            if eos_id is None:
                eos_id = self.weights.config["text_config"].get("eos_token_id")
            for _ in range(max_new_tokens):
                output = self.model(
                    input_ids=next_input,
                    cache=cache,
                    use_cache=True,
                    logits_to_keep=1,
                )
                cache = output.cache
                token_id = int(output.logits[0, -1].argmax().item())
                if eos_id is not None and token_id == int(eos_id):
                    break
                generated.append(token_id)
                yield self.tokenizer.decode(generated, skip_special_tokens=True)
                next_input = self.torch.tensor([[token_id]], dtype=self.torch.long)

    def close(self) -> None:
        self.weights.close()
        if self._tokenizer_tempdir is not None:
            self._tokenizer_tempdir.cleanup()
            self._tokenizer_tempdir = None

    def __enter__(self) -> "StreamedInklingRunner":
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> bool:
        self.close()
        return False
