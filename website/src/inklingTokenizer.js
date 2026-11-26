import { Tokenizer } from "@huggingface/tokenizers";

const MODEL_REPO = "rmaaorg/Inkling-2-Mini-NXS";
const tokenizerPromises = new Map();
const SPECIAL = {
  system: "<|message_system|>",
  user: "<|message_user|>",
  assistant: "<|message_model|>",
  text: "<|content_text|>",
  end: "<|end_message|>",
  endAssistant: "<|content_model_end_sampling|>",
};
const EFFORT = { none: 0, minimal: 0.1, low: 0.2, medium: 0.7, high: 0.9, xhigh: 0.99, max: 0.99 };

async function fetchJson(filename, revision) {
  const url = `https://huggingface.co/${MODEL_REPO}/resolve/${encodeURIComponent(revision)}/${filename}`;
  const response = await fetch(url, { cache: "force-cache" });
  if (!response.ok) throw new Error(`Could not load ${filename} from Hugging Face (HTTP ${response.status}).`);
  return response.json();
}

/** Load only tokenizer metadata in the browser; the Render worker stays tokenizer-free. */
export function loadInklingTokenizer(revision = "main") {
  const pinnedRevision = String(revision).trim();
  if (!/^(?:[a-f0-9]{40}|main)$/i.test(pinnedRevision)) {
    return Promise.reject(new Error("The server reported an invalid Hugging Face model revision."));
  }
  if (!tokenizerPromises.has(pinnedRevision)) {
    const request = Promise.all([
      fetchJson("tokenizer.json", pinnedRevision),
      fetchJson("tokenizer_config.json", pinnedRevision),
    ])
      .then(([tokenizerJson, tokenizerConfig]) => new Tokenizer(tokenizerJson, tokenizerConfig))
      .catch((error) => {
        tokenizerPromises.delete(pinnedRevision);
        throw error;
      });
    tokenizerPromises.set(pinnedRevision, request);
  }
  return tokenizerPromises.get(pinnedRevision);
}

export function renderInklingPrompt(messages, reasoningEffort = "low") {
  const effort = typeof reasoningEffort === "string" ? EFFORT[reasoningEffort.trim()] : Number(reasoningEffort);
  if (effort === undefined || !Number.isFinite(effort) || effort < 0 || effort > 0.99) {
    throw new Error(`Unsupported reasoning effort: ${reasoningEffort}`);
  }

  const effortText = effort === 0 ? "0" : String(effort);
  const parts = [];
  let effortPending = true;
  for (const message of messages) {
    const roleToken = SPECIAL[message.role];
    if (!roleToken) throw new Error(`Unsupported chat role: ${message.role}`);
    if (typeof message.content !== "string") throw new Error("The demo accepts text-only messages.");
    if (effortPending && message.role !== "system") {
      parts.push(`${SPECIAL.system}${SPECIAL.text}Thinking effort level: ${effortText}${SPECIAL.end}`);
      effortPending = false;
    }
    parts.push(`${roleToken}${SPECIAL.text}${message.content}${SPECIAL.end}`);
    if (message.role === "assistant") parts.push(SPECIAL.endAssistant);
  }
  if (effortPending) {
    parts.push(`${SPECIAL.system}${SPECIAL.text}Thinking effort level: ${effortText}${SPECIAL.end}`);
  }
  return parts.join("");
}

/** Encode the same chat template as dotnxs.streamed_inkling, dropping oldest turns if needed. */
export function encodeInklingConversation(tokenizer, messages, maxContextTokens) {
  if (!Number.isInteger(maxContextTokens) || maxContextTokens < 1) {
    throw new Error("The server reported an invalid context limit.");
  }
  let kept = [...messages];
  let ids = tokenizer.encode(renderInklingPrompt(kept)).ids;
  while (ids.length > maxContextTokens && kept.length > 1) {
    const oldestTurn = kept.findIndex((message) => message.role === "user");
    if (oldestTurn < 0) break;
    const removeCount = kept[oldestTurn + 1]?.role === "assistant" ? 2 : 1;
    kept = [...kept.slice(0, oldestTurn), ...kept.slice(oldestTurn + removeCount)];
    ids = tokenizer.encode(renderInklingPrompt(kept)).ids;
  }
  if (ids.length > maxContextTokens) {
    throw new Error(`This message needs ${ids.length} tokens, above the ${maxContextTokens}-token limit. Shorten it and retry.`);
  }
  return ids;
}

export function decodeInklingTokens(tokenizer, tokenIds) {
  return tokenizer.decode(tokenIds, { skip_special_tokens: true });
}

export const INKLING_MAX_CONTEXT_FALLBACK = 64;
