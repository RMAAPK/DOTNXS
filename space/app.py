"""Hugging Face Spaces entry point for the NXS-backed Inkling chat demo."""

from __future__ import annotations

import os

import gradio as gr

from dotnxs.streamed_inkling import StreamedInklingRunner

CACHE_BYTES = 200 * 1024 * 1024
MAX_CONTEXT_TOKENS = 64
MAX_NEW_TOKENS = 12

# One CPU runner is shared by requests. It serializes generation while paging
# each dense layer and selected MoE experts from the single NXS archive.
runner = StreamedInklingRunner.from_hf_repo(
    repo_id="rmaaorg/Inkling-2-Mini-NXS",
    filename="Inkling-2-Mini.nxs",
    cache_bytes=CACHE_BYTES,
    max_context_tokens=MAX_CONTEXT_TOKENS,
)


def respond(message: str, history: list[dict[str, str]]):
    messages = []
    for item in history or []:
        if isinstance(item, dict) and item.get("role") in {"user", "assistant", "system"}:
            content = item.get("content")
            if isinstance(content, str) and content:
                messages.append({"role": item["role"], "content": content})
    messages.append({"role": "user", "content": message})
    yield from runner.stream_chat(messages, max_new_tokens=MAX_NEW_TOKENS)


chat = gr.ChatInterface(
    fn=respond,
    type="messages",
    title="DOTNXS · Inkling 2 Mini",
    description=(
        "Text-only CPU inference from the original single .nxs archive. "
        "The NXS chunk cache is capped at 200 MiB and fills lazily. "
        "This is an experimental streamed runtime; short prompts and slow first tokens are expected."
    ),
    textbox=gr.Textbox(placeholder="Message Inkling…", lines=2, max_lines=4),
    submit_btn="Send",
    stop_btn="Stop",
)

app = gr.Blocks(title="DOTNXS · Inkling Chat")
with app:
    chat.render()

if __name__ == "__main__":
    app.queue(default_concurrency_limit=1, max_size=4).launch(
        server_name="0.0.0.0",
        server_port=int(os.environ.get("PORT", "7860")),
    )
