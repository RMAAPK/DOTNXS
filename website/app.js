const config = window.DOTNXS_CONFIG || {};
const apiBase = String(config.apiBaseUrl || "").replace(/\/$/, "");
const form = document.getElementById("chat-form");
const input = document.getElementById("chat-input");
const sendButton = document.getElementById("send-button");
const stopButton = document.getElementById("stop-button");
const statusPill = document.getElementById("chat-status");
const messagesNode = document.getElementById("chat-messages");
const welcomeNode = document.getElementById("welcome-message");
const history = [];
let activeController = null;

function setStatus(label, state = "idle") {
  statusPill.className = `status-pill ${state}`;
  statusPill.lastChild.textContent = label;
}

function addMessage(role, text = "", { error = false } = {}) {
  const wrapper = document.createElement("div");
  wrapper.className = `message ${role === "user" ? "user-message" : "assistant-message"}${error ? " error-message" : ""}`;
  const label = document.createElement("span");
  label.className = "message-label";
  label.textContent = role === "user" ? "YOU" : error ? "SERVER" : "INKLING";
  const body = document.createElement("p");
  body.textContent = text;
  wrapper.append(label, body);
  messagesNode.append(wrapper);
  messagesNode.scrollTop = messagesNode.scrollHeight;
  return body;
}

function parseSseBlock(block) {
  let event = "message";
  const data = [];
  for (const line of block.split(/\r?\n/)) {
    if (line.startsWith("event:")) event = line.slice(6).trim();
    else if (line.startsWith("data:")) data.push(line.slice(5).trimStart());
  }
  if (!data.length) return null;
  return { event, payload: JSON.parse(data.join("\n")) };
}

async function checkHealth() {
  if (!apiBase) {
    setStatus("API URL not configured", "fail");
    welcomeNode.textContent = "Set the Render API URL in website/config.js to enable chat.";
    return;
  }
  try {
    const response = await fetch(`${apiBase}/health`, { cache: "no-store" });
    const health = await response.json();
    if (!response.ok || health.status === "error") throw new Error(health.detail || "API health check failed");
    if (health.status === "ready") {
      setStatus("CPU server ready", "pass");
      welcomeNode.textContent = "Send a short message to Inkling. The CPU server streams the original NXS weights as needed.";
    } else {
      setStatus("Warming up model", "busy");
      welcomeNode.textContent = "The Render server is loading tokenizer and model metadata. Chat will enable when it is ready.";
      window.setTimeout(checkHealth, 15000);
    }
  } catch (error) {
    setStatus("Server unavailable", "fail");
    welcomeNode.textContent = `Render API is not reachable yet: ${error.message}`;
    window.setTimeout(checkHealth, 20000);
  }
}

async function readStream(response, bodyNode) {
  if (!response.body) throw new Error("The server did not return a streaming response.");
  const reader = response.body.getReader();
  const decoder = new TextDecoder();
  let buffer = "";
  let assistantText = "";

  const consume = (block) => {
    const parsed = parseSseBlock(block);
    if (!parsed) return;
    if (parsed.event === "token") {
      assistantText = parsed.payload.text || "";
      bodyNode.textContent = assistantText || "…";
      messagesNode.scrollTop = messagesNode.scrollHeight;
    } else if (parsed.event === "error") {
      throw new Error(parsed.payload.message || "Model generation failed.");
    }
  };

  while (true) {
    const { value, done } = await reader.read();
    buffer += decoder.decode(value || new Uint8Array(), { stream: !done });
    let splitAt;
    while ((splitAt = buffer.indexOf("\n\n")) >= 0) {
      const block = buffer.slice(0, splitAt);
      buffer = buffer.slice(splitAt + 2);
      consume(block);
    }
    if (done) break;
  }
  if (buffer.trim()) consume(buffer);
  return assistantText;
}

form.addEventListener("submit", async (event) => {
  event.preventDefault();
  const text = input.value.trim();
  if (!text || !apiBase || activeController) return;

  const userMessage = { role: "user", content: text };
  history.push(userMessage);
  addMessage("user", text);
  const assistantBody = addMessage("assistant", "Waiting for the CPU model…");
  activeController = new AbortController();
  input.value = "";
  input.disabled = true;
  sendButton.disabled = true;
  stopButton.disabled = false;
  setStatus("Generating on CPU", "busy");

  try {
    const response = await fetch(`${apiBase}/v1/chat`, {
      method: "POST",
      headers: { "Content-Type": "application/json", Accept: "text/event-stream" },
      body: JSON.stringify({ messages: history }),
      signal: activeController.signal,
    });
    if (!response.ok) {
      let detail = `Render API returned HTTP ${response.status}`;
      try {
        const payload = await response.json();
        detail = payload.detail || detail;
      } catch (_) {}
      throw new Error(detail);
    }
    const answer = await readStream(response, assistantBody);
    if (answer) history.push({ role: "assistant", content: answer });
    else assistantBody.textContent = "(No text was generated.)";
    setStatus("CPU server ready", "pass");
  } catch (error) {
    if (error.name === "AbortError") {
      assistantBody.textContent = "Generation stopped.";
      setStatus("Stopped", "idle");
    } else {
      assistantBody.textContent = error.message;
      assistantBody.parentElement.classList.add("error-message");
      setStatus("Generation failed", "fail");
    }
  } finally {
    activeController = null;
    input.disabled = false;
    sendButton.disabled = false;
    stopButton.disabled = true;
    input.focus();
  }
});

stopButton.addEventListener("click", () => activeController?.abort());
checkHealth();
