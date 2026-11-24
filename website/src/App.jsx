import React, { useCallback, useEffect, useRef, useState } from "react";
import {
  ActivityIndicator,
  Pressable,
  ScrollView,
  StyleSheet,
  Text,
  TextInput,
  View,
  useWindowDimensions,
} from "react-native";

const API_BASE = String(
  window.DOTNXS_CONFIG?.apiBaseUrl || "https://dotnxs-inkling-api.onrender.com",
).replace(/\/$/, "");
const COLORS = {
  background: "#080c18",
  panel: "#11182b",
  border: "rgba(164,179,220,0.15)",
  text: "#f2f5ff",
  muted: "#98a3bd",
  cyan: "#64e5da",
  violet: "#a18aff",
  red: "#ff728c",
  green: "#69e3a5",
};

function parseEvent(block) {
  let event = "message";
  const data = [];
  for (const line of block.split(/\r?\n/)) {
    if (line.startsWith("event:")) event = line.slice(6).trim();
    else if (line.startsWith("data:")) data.push(line.slice(5).trimStart());
  }
  return data.length ? { event, payload: JSON.parse(data.join("\n")) } : null;
}

export default function App() {
  const { width } = useWindowDimensions();
  const [serverStatus, setServerStatus] = useState("Checking Render server");
  const [ready, setReady] = useState(false);
  const [messages, setMessages] = useState([
    {
      role: "assistant",
      content: "Connecting to the Render CPU server. The first response may take a while while NXS weights are fetched.",
      welcome: true,
    },
  ]);
  const [conversation, setConversation] = useState([]);
  const [draft, setDraft] = useState("");
  const [busy, setBusy] = useState(false);
  const abortRef = useRef(null);
  const scrollRef = useRef(null);
  const wide = width >= 820;

  const checkHealth = useCallback(async () => {
    try {
      const response = await fetch(`${API_BASE}/health`, { cache: "no-store" });
      const health = await response.json();
      if (!response.ok || health.status === "error") {
        throw new Error(health.detail || "Render API is unavailable");
      }
      const isReady = health.status === "ready";
      setReady(isReady);
      setServerStatus(isReady ? "CPU server ready" : "Warming up model");
    } catch (error) {
      setReady(false);
      setServerStatus("Server unavailable");
      setMessages((previous) => {
        const index = previous.findIndex((message) => message.welcome);
        if (index < 0) return previous;
        const next = [...previous];
        next[index] = {
          ...next[index],
          content: `Render API is unavailable: ${error.message}. Check the server memory limit and startup status.`,
        };
        return next;
      });
    }
  }, []);

  useEffect(() => {
    checkHealth();
    const timer = window.setInterval(checkHealth, 15000);
    return () => window.clearInterval(timer);
  }, [checkHealth]);

  useEffect(() => {
    scrollRef.current?.scrollToEnd({ animated: true });
  }, [messages]);

  const stop = () => abortRef.current?.abort();

  const send = async () => {
    const content = draft.trim();
    if (!content || busy || !ready) return;

    const userMessage = { role: "user", content };
    const requestMessages = [...conversation, userMessage];
    setConversation(requestMessages);
    setMessages((previous) => [
      ...previous.filter((message) => !message.welcome),
      userMessage,
      { role: "assistant", content: "Fetching model weights…", pending: true },
    ]);
    setDraft("");
    setBusy(true);
    const controller = new AbortController();
    abortRef.current = controller;

    try {
      const response = await fetch(`${API_BASE}/v1/chat`, {
        method: "POST",
        headers: { "Content-Type": "application/json", Accept: "text/event-stream" },
        body: JSON.stringify({ messages: requestMessages }),
        signal: controller.signal,
      });
      if (!response.ok) {
        let detail = `Render API returned HTTP ${response.status}`;
        try {
          const body = await response.json();
          detail = body.detail || detail;
        } catch (_) {}
        throw new Error(detail);
      }
      if (!response.body) throw new Error("The API returned no response stream.");

      const reader = response.body.getReader();
      const decoder = new TextDecoder();
      let buffer = "";
      let answer = "";
      while (true) {
        const { value, done: finished } = await reader.read();
        buffer += decoder.decode(value || new Uint8Array(), { stream: !finished });
        let splitAt;
        while ((splitAt = buffer.indexOf("\n\n")) >= 0) {
          const block = buffer.slice(0, splitAt);
          buffer = buffer.slice(splitAt + 2);
          const parsed = parseEvent(block);
          if (!parsed) continue;
          if (parsed.event === "token") {
            answer = parsed.payload.text || "";
            setMessages((previous) => {
              const next = [...previous];
              const currentIndex = next.findIndex((message) => message.pending);
              if (currentIndex >= 0) next[currentIndex] = { role: "assistant", content: answer };
              return next;
            });
          } else if (parsed.event === "error") {
            throw new Error(parsed.payload.message || "Model inference failed.");
          }
        }
        if (finished) break;
      }
      if (buffer.trim()) {
        const parsed = parseEvent(buffer);
        if (parsed?.event === "token") answer = parsed.payload.text || "";
        if (parsed?.event === "error") throw new Error(parsed.payload.message || "Model inference failed.");
      }
      if (answer) {
        setConversation([...requestMessages, { role: "assistant", content: answer }]);
        setMessages((previous) => {
          const next = [...previous];
          const currentIndex = next.findIndex((message) => message.pending);
          if (currentIndex >= 0) next[currentIndex] = { role: "assistant", content: answer };
          return next;
        });
      } else {
        setMessages((previous) => {
          const next = [...previous];
          const currentIndex = next.findIndex((message) => message.pending);
          if (currentIndex >= 0) next[currentIndex] = { role: "assistant", content: "(No text was generated.)" };
          return next;
        });
      }
      setReady(true);
      setServerStatus("CPU server ready");
    } catch (error) {
      const message = error.name === "AbortError" ? "Generation stopped." : error.message;
      setMessages((previous) => {
        const next = [...previous];
        const currentIndex = next.findIndex((item) => item.pending);
        if (currentIndex >= 0) next[currentIndex] = { role: "assistant", content: message, error: true };
        return next;
      });
      if (error.name !== "AbortError") {
        setReady(false);
        setServerStatus("Request failed");
      }
    } finally {
      abortRef.current = null;
      setBusy(false);
    }
  };

  return (
    <View style={styles.page}>
      <View style={[styles.container, wide && styles.containerWide]}>
        <View style={styles.header}>
          <Pressable onPress={() => (window.location.href = "https://github.com/RMAAPK/DOTNXS")} style={styles.brand}>
            <View style={styles.brandMark}><Text style={styles.brandMarkText}>N</Text></View>
            <View>
              <Text style={styles.brandName}>DOTNXS</Text>
              <Text style={styles.brandSub}>INKLING CHAT</Text>
            </View>
          </Pressable>
          <View style={styles.nav}>
            <Pressable onPress={() => (window.location.href = "https://huggingface.co/rmaaorg/Inkling-2-Mini-NXS")}>
              <Text style={styles.navText}>Model archive</Text>
            </Pressable>
            <Pressable onPress={() => (window.location.href = "/ping.html")}>
              <Text style={styles.navText}>Ping server</Text>
            </Pressable>
          </View>
        </View>

        <View style={styles.hero}>
          <Text style={styles.eyebrow}>CPU · NXS-STREAMED MODEL</Text>
          <Text style={styles.title}>Chat with Inkling.</Text>
          <Text style={styles.subtitle}>Weights page on demand.</Text>
          <Text style={styles.heroCopy}>
            The original checkpoint stays in one .nxs archive. A C++ Render server fronts the CPU model worker; dense layers
            and selected NVFP4/MXFP8 experts are loaded only as needed.
          </Text>
          <View style={styles.chips}>
            <Text style={styles.chip}>single .nxs</Text>
            <Text style={styles.chip}>C++ API · CPU worker</Text>
            <Text style={styles.chip}>200 MiB NXS cache</Text>
            <Text style={styles.chip}>Apache-2.0 model</Text>
          </View>
        </View>

        <View style={[styles.content, wide && styles.contentWide]}>
          <View style={styles.chatPanel}>
            <View style={styles.panelHeader}>
              <View>
                <Text style={styles.eyebrow}>MODEL SESSION</Text>
                <Text style={styles.panelTitle}>Inkling 2 Mini</Text>
              </View>
              <View style={[styles.status, ready ? styles.statusReady : styles.statusPending]}>
                {!ready && <ActivityIndicator size="small" color={COLORS.cyan} />}
                <Text style={styles.statusText}>{serverStatus}</Text>
              </View>
            </View>

            <ScrollView
              ref={scrollRef}
              style={styles.messages}
              contentContainerStyle={styles.messagesContent}
              onContentSizeChange={() => scrollRef.current?.scrollToEnd({ animated: true })}
            >
              {messages.map((message, index) => (
                <View
                  key={`${message.role}-${index}`}
                  style={[
                    styles.bubble,
                    message.role === "user" ? styles.userBubble : styles.assistantBubble,
                    message.error && styles.errorBubble,
                  ]}
                >
                  <Text style={styles.messageLabel}>{message.role === "user" ? "YOU" : message.error ? "SERVER" : "INKLING"}</Text>
                  <Text style={styles.messageText}>{message.content}</Text>
                  {message.pending && <ActivityIndicator style={styles.inlineSpinner} size="small" color={COLORS.cyan} />}
                </View>
              ))}
            </ScrollView>

            <View style={styles.composer}>
              <TextInput
                value={draft}
                onChangeText={setDraft}
                onSubmitEditing={send}
                editable={ready && !busy}
                placeholder={ready ? "Message Inkling…" : "Waiting for the Render server…"}
                placeholderTextColor="#75809a"
                multiline
                style={styles.input}
              />
              <Pressable onPress={send} disabled={!ready || busy || !draft.trim()} style={[styles.sendButton, (!ready || busy || !draft.trim()) && styles.buttonDisabled]}>
                <Text style={styles.sendText}>Send</Text>
              </Pressable>
              <Pressable onPress={stop} disabled={!busy} style={[styles.stopButton, !busy && styles.buttonDisabled]}>
                <Text style={styles.stopText}>Stop</Text>
              </Pressable>
            </View>
            <Text style={styles.footnote}>Short prompts recommended · CPU streaming can be slow · 200 MiB archive cache</Text>
          </View>

          <View style={styles.sideColumn}>
            <View style={styles.sidePanel}>
              <Text style={styles.eyebrow}>TARGET MODEL</Text>
              <Text style={styles.sideTitle}>Inkling 2 Mini</Text>
              <View style={styles.statRow}><Text style={styles.statValue}>26.4B</Text><Text style={styles.statLabel}>total parameters</Text></View>
              <View style={styles.statRow}><Text style={styles.statValue}>3.4B</Text><Text style={styles.statLabel}>active parameters</Text></View>
              <Text style={styles.sideHint}>The Render service reads the public NXS archive directly and never stages the full checkpoint in RAM.</Text>
              <Pressable onPress={() => (window.location.href = "https://huggingface.co/rmaaorg/Inkling-2-Mini-NXS")}>
                <Text style={styles.linkText}>Open model archive ↗</Text>
              </Pressable>
            </View>
            <View style={styles.sidePanel}>
              <Text style={styles.eyebrow}>MEMORY POLICY</Text>
              <Text style={styles.sideTitle}>200 MiB NXS cache</Text>
              <Text style={styles.sideHint}>Verified chunks enter a lazy LRU cache. This is a cache limit, not the server's total RAM use.</Text>
            </View>
          </View>
        </View>

        <View style={styles.footer}>
          <Text style={styles.footerText}>DOTNXS · Render C++ API and streamed CPU worker</Text>
          <Text style={styles.footerText}>Inkling 2 Mini weights: Apache-2.0</Text>
        </View>
      </View>
    </View>
  );
}

const styles = StyleSheet.create({
  page: { flex: 1, minHeight: "100vh", backgroundColor: COLORS.background, paddingVertical: 18 },
  container: { width: "100%", maxWidth: 1120, alignSelf: "center", paddingHorizontal: 20 },
  containerWide: { paddingHorizontal: 28 },
  header: { minHeight: 64, flexDirection: "row", alignItems: "center", justifyContent: "space-between", borderBottomWidth: 1, borderBottomColor: COLORS.border },
  brand: { flexDirection: "row", alignItems: "center", gap: 11 },
  brandMark: { width: 36, height: 36, borderRadius: 11, borderWidth: 1, borderColor: COLORS.violet, alignItems: "center", justifyContent: "center", backgroundColor: "#17152c" },
  brandMarkText: { color: COLORS.cyan, fontSize: 18, fontWeight: "800" },
  brandName: { color: COLORS.text, fontSize: 13, fontWeight: "800", letterSpacing: 1.5 },
  brandSub: { color: COLORS.muted, fontSize: 9, letterSpacing: 2 },
  nav: { flexDirection: "row", alignItems: "center", gap: 18 },
  navText: { color: COLORS.cyan, fontSize: 12, fontWeight: "700" },
  hero: { paddingTop: 42, paddingBottom: 28 },
  eyebrow: { color: COLORS.cyan, fontSize: 10, letterSpacing: 2, fontWeight: "700" },
  title: { marginTop: 14, color: COLORS.text, fontSize: 48, fontWeight: "800", letterSpacing: -2 },
  subtitle: { marginTop: 2, color: COLORS.violet, fontSize: 24, fontWeight: "700" },
  heroCopy: { maxWidth: 760, marginTop: 14, color: "#aab4cc", fontSize: 14, lineHeight: 22 },
  chips: { flexDirection: "row", flexWrap: "wrap", gap: 8, marginTop: 16 },
  chip: { color: "#c4cde0", backgroundColor: "rgba(255,255,255,0.035)", borderColor: COLORS.border, borderWidth: 1, borderRadius: 18, paddingHorizontal: 10, paddingVertical: 7, fontSize: 10 },
  content: { flexDirection: "column", gap: 16, alignItems: "stretch" },
  contentWide: { flexDirection: "row", alignItems: "flex-start" },
  chatPanel: { flex: 1.55, minHeight: 480, padding: 20, borderColor: COLORS.border, borderWidth: 1, borderRadius: 18, backgroundColor: COLORS.panel },
  panelHeader: { flexDirection: "row", alignItems: "flex-start", justifyContent: "space-between", gap: 12, marginBottom: 16 },
  panelTitle: { marginTop: 6, color: COLORS.text, fontSize: 20, fontWeight: "700" },
  status: { flexDirection: "row", alignItems: "center", gap: 7, borderWidth: 1, borderColor: COLORS.border, borderRadius: 16, paddingHorizontal: 10, paddingVertical: 7 },
  statusReady: { borderColor: "rgba(105,227,165,0.4)" },
  statusPending: { borderColor: "rgba(255,209,102,0.3)" },
  statusText: { color: COLORS.muted, fontSize: 10 },
  messages: { height: 340, borderBottomWidth: 1, borderBottomColor: COLORS.border, marginBottom: 14 },
  messagesContent: { gap: 10, paddingVertical: 8 },
  bubble: { maxWidth: "88%", padding: 13, borderRadius: 14, borderWidth: 1, borderColor: COLORS.border },
  userBubble: { alignSelf: "flex-end", backgroundColor: "rgba(161,138,255,0.12)", borderBottomRightRadius: 4 },
  assistantBubble: { alignSelf: "flex-start", backgroundColor: "rgba(255,255,255,0.035)", borderBottomLeftRadius: 4 },
  errorBubble: { borderColor: "rgba(255,114,140,0.4)" },
  messageLabel: { color: COLORS.cyan, fontSize: 9, letterSpacing: 1.4, fontWeight: "700" },
  messageText: { marginTop: 7, color: "#c0cbe0", fontSize: 12, lineHeight: 19 },
  inlineSpinner: { marginTop: 8, alignSelf: "flex-start" },
  composer: { flexDirection: "row", alignItems: "flex-end", gap: 8 },
  input: { flex: 1, minHeight: 54, maxHeight: 140, padding: 12, color: COLORS.text, backgroundColor: "#090e1b", borderColor: COLORS.border, borderWidth: 1, borderRadius: 10, fontSize: 12, textAlignVertical: "top" },
  sendButton: { minHeight: 42, paddingHorizontal: 16, alignItems: "center", justifyContent: "center", borderRadius: 10, backgroundColor: COLORS.cyan },
  sendText: { color: "#071318", fontSize: 12, fontWeight: "800" },
  stopButton: { minHeight: 42, paddingHorizontal: 14, alignItems: "center", justifyContent: "center", borderRadius: 10, borderWidth: 1, borderColor: COLORS.border, backgroundColor: "rgba(255,255,255,0.04)" },
  stopText: { color: COLORS.text, fontSize: 12, fontWeight: "700" },
  buttonDisabled: { opacity: 0.45 },
  footnote: { marginTop: 9, color: "#78839e", fontSize: 9 },
  sideColumn: { flex: 0.8, gap: 14 },
  sidePanel: { padding: 18, borderColor: COLORS.border, borderWidth: 1, borderRadius: 16, backgroundColor: COLORS.panel },
  sideTitle: { marginTop: 7, color: COLORS.text, fontSize: 17, fontWeight: "700" },
  statRow: { flexDirection: "row", alignItems: "baseline", gap: 10, paddingVertical: 11, borderBottomWidth: 1, borderBottomColor: COLORS.border },
  statValue: { minWidth: 72, color: COLORS.text, fontSize: 16, fontWeight: "700" },
  statLabel: { color: COLORS.muted, fontSize: 11 },
  sideHint: { marginTop: 12, marginBottom: 10, color: COLORS.muted, fontSize: 11, lineHeight: 17 },
  linkText: { color: COLORS.cyan, fontSize: 11, fontWeight: "700" },
  footer: { flexDirection: "row", justifyContent: "space-between", flexWrap: "wrap", gap: 10, paddingVertical: 22 },
  footerText: { color: "#68738d", fontSize: 9 },
});
