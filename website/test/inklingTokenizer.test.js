import assert from "node:assert/strict";
import test from "node:test";
import {
  decodeInklingTokens,
  encodeInklingConversation,
  loadInklingTokenizer,
  renderInklingPrompt,
} from "../src/inklingTokenizer.js";

const user = (content) => ({ role: "user", content });
const assistant = (content) => ({ role: "assistant", content });

test("renders the Inkling low-effort prompt template", () => {
  assert.equal(
    renderInklingPrompt([user("hello")]),
    "<|message_system|><|content_text|>Thinking effort level: 0.2<|end_message|>" +
      "<|message_user|><|content_text|>hello<|end_message|>",
  );
});

test("marks completed assistant turns with the model end token", () => {
  const prompt = renderInklingPrompt([user("hello"), assistant("hi"), user("again")]);
  assert.ok(prompt.includes("<|message_model|><|content_text|>hi<|end_message|><|content_model_end_sampling|>"));
  assert.ok(prompt.endsWith("<|message_user|><|content_text|>again<|end_message|>"));
});

test("drops oldest complete turns when the bounded context is exceeded", () => {
  const encodedPrompts = [];
  const fakeTokenizer = {
    encode(prompt) {
      encodedPrompts.push(prompt);
      return { ids: Array(Math.ceil(prompt.length / 8)).fill(1) };
    },
  };
  const ids = encodeInklingConversation(
    fakeTokenizer,
    [user("old message ".repeat(8)), assistant("old answer ".repeat(8)), user("latest")],
    20,
  );
  assert.ok(ids.length <= 20);
  assert.ok(!encodedPrompts.at(-1).includes("old message"));
  assert.ok(encodedPrompts.at(-1).includes("latest"));
});

test("rejects a single message that cannot fit the context", () => {
  const fakeTokenizer = { encode: (prompt) => ({ ids: Array(prompt.length).fill(0) }) };
  assert.throws(() => encodeInklingConversation(fakeTokenizer, [user("too long")], 10), /above the 10-token limit/);
});

test("decodes generated IDs while removing special tokens", () => {
  const fakeTokenizer = { decode: (ids, options) => `${ids.join(",")}:${options.skip_special_tokens}` };
  assert.equal(decodeInklingTokens(fakeTokenizer, [21, 22]), "21,22:true");
});

test("requires a resolved model commit before loading tokenizer assets", async () => {
  await assert.rejects(loadInklingTokenizer("not-a-commit"), /invalid Hugging Face model revision/);
});
