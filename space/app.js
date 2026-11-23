const MAGIC = Uint8Array.from([
  0x4e, 0x58, 0x53, 0x00, 0x62, 0x62, 0x96, 0x6d, 0x2f, 0x13,
  0x49, 0xc9, 0x86, 0x91, 0xc5, 0x22, 0xdc, 0xe3, 0x08, 0x43,
]);
const HEADER_BYTES = 80;
const MAX_MANIFEST_BYTES = 128 * 1024 * 1024;
const DEFAULT_ARCHIVE_URL =
  "https://huggingface.co/rmaaorg/Inkling-2-Mini-NXS/resolve/main/Inkling-2-Mini.nxs";

const $ = (id) => document.getElementById(id);
const elements = {
  url: $("archive-url"),
  button: $("probe-button"),
  status: $("probe-status"),
  error: $("probe-error"),
  results: $("probe-results"),
  archiveSize: $("archive-size"),
  entryCount: $("entry-count"),
  requestCount: $("request-count"),
  bytesFetched: $("bytes-fetched"),
  verifiedMessage: $("verified-message"),
  sampleName: $("sample-name"),
  sampleSize: $("sample-size"),
  log: $("request-log"),
  elapsed: $("elapsed"),
  gpuStatus: $("gpu-status"),
  gpuDetails: $("gpu-details"),
};

let bytesFetched = 0;
let rangeRequests = 0;
let startedAt = 0;

function formatBytes(value) {
  if (!Number.isFinite(value)) return "unknown";
  const units = ["B", "KB", "MB", "GB", "TB"];
  let size = value;
  let unit = 0;
  while (size >= 1024 && unit < units.length - 1) {
    size /= 1024;
    unit += 1;
  }
  return `${size.toFixed(unit === 0 ? 0 : 2)} ${units[unit]}`;
}

function hex(bytes) {
  return Array.from(bytes, (byte) => byte.toString(16).padStart(2, "0")).join("");
}

function addLog(message, state = "info") {
  const placeholder = elements.log.querySelector(".log-placeholder");
  if (placeholder) placeholder.remove();
  const item = document.createElement("li");
  item.className = `log-${state}`;
  item.textContent = message;
  elements.log.append(item);
  elements.log.scrollTop = elements.log.scrollHeight;
}

function setStatus(label, state) {
  elements.status.className = `status-pill ${state}`;
  elements.status.querySelector(".status-dot").setAttribute("aria-hidden", "true");
  elements.status.lastChild.textContent = label;
}

function getReadAheadBytes() {
  // A small probe is deliberate: it should demonstrate selective I/O, not warm the whole model.
  return 256 * 1024;
}

async function fetchRange(url, start, length, totalSize = null) {
  if (!Number.isSafeInteger(start) || !Number.isSafeInteger(length) || start < 0 || length <= 0) {
    throw new Error("Invalid NXS byte range.");
  }
  const requestedLength = Math.max(length, getReadAheadBytes());
  const endExclusive = totalSize === null ? start + requestedLength : Math.min(totalSize, start + requestedLength);
  const end = endExclusive - 1;
  const response = await fetch(url, {
    method: "GET",
    mode: "cors",
    credentials: "omit",
    cache: "no-store",
    headers: { Range: `bytes=${start}-${end}` },
  });
  if (response.status !== 206) {
    if (response.body) await response.body.cancel();
    throw new Error(
      `Range request was not honored (HTTP ${response.status}); refusing a full-file download. ` +
      "Check the Hub gateway's CORS and Range support."
    );
  }

  const contentRange = response.headers.get("Content-Range");
  let total = totalSize;
  if (contentRange) {
    const match = /^bytes (\d+)-(\d+)\/(\d+|\*)$/i.exec(contentRange);
    if (!match || Number(match[1]) !== start) {
      await response.body?.cancel();
      throw new Error(`Unexpected Content-Range header: ${contentRange}`);
    }
    const actualEnd = Number(match[2]);
    if (actualEnd > end || actualEnd < start) {
      await response.body?.cancel();
      throw new Error(`Invalid Content-Range bounds: ${contentRange}`);
    }
    if (match[3] !== "*") {
      const responseTotal = Number(match[3]);
      if (total !== null && total !== responseTotal) {
        await response.body?.cancel();
        throw new Error("Archive size changed during the probe.");
      }
      total = responseTotal;
    }
  }

  const bytes = new Uint8Array(await response.arrayBuffer());
  const expectedLength = endExclusive - start;
  if (bytes.byteLength !== expectedLength || bytes.byteLength < length) {
    throw new Error(`Short range response: requested ${expectedLength} bytes, received ${bytes.byteLength}.`);
  }
  bytesFetched += bytes.byteLength;
  rangeRequests += 1;
  return { bytes, total, start };
}

async function sha256(bytes) {
  const digest = await crypto.subtle.digest("SHA-256", bytes);
  return hex(new Uint8Array(digest));
}

function parseHeader(bytes) {
  if (bytes.byteLength < HEADER_BYTES) throw new Error("NXS header is truncated.");
  for (let i = 0; i < MAGIC.length; i += 1) {
    if (bytes[i] !== MAGIC[i]) throw new Error("NXS magic does not match DOTNXS.");
  }
  const view = new DataView(bytes.buffer, bytes.byteOffset, bytes.byteLength);
  const major = view.getUint16(20, true);
  const minor = view.getUint16(22, true);
  const headerSize = view.getUint32(24, true);
  const flags = view.getUint32(28, true);
  const manifestOffset = Number(view.getBigUint64(32, true));
  const manifestLength = Number(view.getBigUint64(40, true));
  const manifestHash = hex(bytes.subarray(48, 80));
  if (major !== 0 || minor !== 2 || headerSize !== HEADER_BYTES || flags !== 0) {
    throw new Error(`Unsupported NXS header ${major}.${minor} (size ${headerSize}, flags ${flags}).`);
  }
  if (!Number.isSafeInteger(manifestOffset) || !Number.isSafeInteger(manifestLength) || manifestLength <= 0) {
    throw new Error("NXS manifest range is invalid.");
  }
  if (manifestLength > MAX_MANIFEST_BYTES) throw new Error("Manifest exceeds the demo's safety limit.");
  return { major, minor, manifestOffset, manifestLength, manifestHash };
}

function readTensorEntry(manifest) {
  if (!manifest || manifest.format !== "DOTNXS" || !Array.isArray(manifest.entries)) {
    throw new Error("Invalid DOTNXS manifest.");
  }
  const entry = manifest.entries.find(
    (candidate) => candidate.kind === "tensor" && Array.isArray(candidate.chunks) && candidate.chunks.length > 0
  );
  if (!entry) throw new Error("Manifest contains no tensor chunks to probe.");
  return entry;
}

async function runProbe() {
  const url = elements.url.value.trim();
  if (!url) throw new Error("Enter a public NXS archive URL.");
  bytesFetched = 0;
  rangeRequests = 0;
  startedAt = performance.now();
  elements.results.hidden = true;
  elements.error.hidden = true;
  elements.error.textContent = "";
  elements.log.replaceChildren();
  elements.button.disabled = true;
  elements.button.textContent = "Streaming ranges…";
  setStatus("Connecting", "busy");
  addLog("GET Range: NXS fixed header", "info");

  try {
    const headerRange = await fetchRange(url, 0, HEADER_BYTES);
    const header = parseHeader(headerRange.bytes);
    if (headerRange.total !== null && header.manifestOffset + header.manifestLength > headerRange.total) {
      throw new Error("Manifest points outside the remote archive size.");
    }
    addLog(`Header OK · NXS ${header.major}.${header.minor} · ${formatBytes(headerRange.total ?? 0)} archive`, "ok");

    addLog(`GET Range: manifest @ ${header.manifestOffset} (${formatBytes(header.manifestLength)})`, "info");
    const manifestRange = await fetchRange(url, header.manifestOffset, header.manifestLength, headerRange.total);
    const manifestBytes = manifestRange.bytes.subarray(0, header.manifestLength);
    const actualManifestHash = await sha256(manifestBytes);
    if (actualManifestHash !== header.manifestHash) throw new Error("Manifest SHA-256 mismatch.");
    const manifest = JSON.parse(new TextDecoder().decode(manifestBytes));
    if (manifest.version?.[0] !== 0 || manifest.version?.[1] !== 2) {
      throw new Error(`Unsupported manifest version: ${JSON.stringify(manifest.version)}.`);
    }
    addLog(`Manifest verified · ${manifest.entries.length} entries · SHA-256 OK`, "ok");

    const entry = readTensorEntry(manifest);
    const chunk = entry.chunks[0];
    if (!Number.isSafeInteger(chunk.offset) || !Number.isSafeInteger(chunk.length) || chunk.length <= 0) {
      throw new Error("First tensor chunk has an invalid byte range.");
    }
    addLog(`GET Range: tensor ${entry.name} @ ${chunk.offset} (${formatBytes(chunk.length)})`, "info");
    const chunkRange = await fetchRange(url, chunk.offset, chunk.length, headerRange.total);
    const chunkBytes = chunkRange.bytes.subarray(0, chunk.length);
    const actualChunkHash = await sha256(chunkBytes);
    if (actualChunkHash !== chunk.sha256) throw new Error(`Tensor chunk SHA-256 mismatch for ${entry.name}.`);
    addLog(`Tensor chunk verified · ${formatBytes(chunk.length)} · SHA-256 OK`, "ok");

    const totalSeconds = ((performance.now() - startedAt) / 1000).toFixed(1);
    elements.archiveSize.textContent = formatBytes(headerRange.total ?? 0);
    elements.entryCount.textContent = manifest.entries.length.toLocaleString();
    elements.requestCount.textContent = String(rangeRequests);
    elements.bytesFetched.textContent = formatBytes(bytesFetched);
    elements.sampleName.textContent = entry.name;
    elements.sampleSize.textContent = formatBytes(chunk.length);
    elements.verifiedMessage.textContent = "Header, manifest, and real tensor bytes verified";
    elements.results.hidden = false;
    elements.elapsed.textContent = `${totalSeconds} s`;
    setStatus("Range test passed", "pass");
  } catch (error) {
    const message = error instanceof Error ? error.message : String(error);
    addLog(message, "fail");
    elements.error.textContent = message;
    elements.error.hidden = false;
    setStatus("Range test failed", "fail");
  } finally {
    elements.button.disabled = false;
    elements.button.innerHTML = 'Run range test <span aria-hidden="true">↗</span>';
  }
}

async function checkWebGPU() {
  if (!navigator.gpu) {
    elements.gpuStatus.className = "gpu-state no";
    elements.gpuStatus.textContent = "WebGPU unavailable in this browser";
    elements.gpuDetails.textContent = "Range streaming can still be tested, but this device cannot run the planned WebGPU backend here.";
    return;
  }
  try {
    const adapter = await navigator.gpu.requestAdapter({ powerPreference: "high-performance" });
    if (!adapter) throw new Error("No WebGPU adapter returned.");
    const device = await adapter.requestDevice();
    const maxBuffer = device.limits.maxBufferSize;
    device.destroy();
    elements.gpuStatus.className = "gpu-state ok";
    elements.gpuStatus.textContent = "WebGPU device available";
    elements.gpuDetails.textContent = `Capability check only · max single buffer ${formatBytes(maxBuffer)} · no model weights allocated.`;
  } catch (error) {
    elements.gpuStatus.className = "gpu-state no";
    elements.gpuStatus.textContent = "WebGPU adapter unavailable";
    elements.gpuDetails.textContent = error instanceof Error ? error.message : String(error);
  }
}

elements.button.addEventListener("click", runProbe);
checkWebGPU();
