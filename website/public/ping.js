const apiBase = String(window.DOTNXS_CONFIG?.apiBaseUrl || "").replace(/\/$/, "");
const result = document.getElementById("result");
const button = document.getElementById("ping-button");
const auto = document.getElementById("auto-ping");
document.getElementById("endpoint").textContent = `${apiBase}/ping`;

async function ping() {
  result.textContent = "Pinging Render…";
  try {
    const start = performance.now();
    const response = await fetch(`${apiBase}/ping`, { cache: "no-store" });
    const body = (await response.text()).trim();
    if (!response.ok || body !== "pong") throw new Error(`HTTP ${response.status}: ${body}`);
    result.textContent = `pong · ${(performance.now() - start).toFixed(0)} ms · ${new Date().toLocaleTimeString()}`;
  } catch (error) {
    result.textContent = `Ping failed: ${error.message}`;
  }
}

button.addEventListener("click", ping);
window.setInterval(() => {
  if (auto.checked) ping();
}, 10 * 60 * 1000);
ping();
