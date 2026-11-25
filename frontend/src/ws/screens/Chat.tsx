import { useEffect, useRef, useState } from "react";
import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";
import { Send } from "lucide-react";
import { sendChat, listChats, loadChat, BRIDGE_START,
         type ChatSummary } from "../../lib/chatClient";
import { useDocumentTitle } from "../../lib/useDocumentTitle";

interface Msg { role: "user" | "assistant"; text: string }

export default function Chat() {
  useDocumentTitle("Chat");
  const [messages, setMessages] = useState<Msg[]>([]);
  const [input, setInput] = useState("");
  const [sessionId, setSessionId] = useState<string | null>(null);
  // null = the bridge didn't answer, so history availability is unknown
  const [history, setHistory] = useState<ChatSummary[] | null>([]);
  const [showHistory, setShowHistory] = useState(false);
  const [waiting, setWaiting] = useState(false);
  const endRef = useRef<HTMLDivElement>(null);
  // bumped by every conversation switch (restore, open, new); async results
  // captured before the bump are dropped so they cannot clobber the newer view
  const genRef = useRef(0);

  useEffect(() => {
    endRef.current?.scrollIntoView?.({ behavior: "smooth" });
  }, [messages, waiting]);

  useEffect(() => {
    void restoreLatest();
  }, []);

  async function restoreLatest() {
    const gen = ++genRef.current;
    const chats = await listChats();
    // history is global, not per-conversation: a result that races a
    // newer conversation switch still fills the list correctly, so it
    // lands above the generation guard (which protects messages/sessionId)
    setHistory(chats);
    if (genRef.current !== gen) return;
    if (!chats || chats.length === 0) return;
    const saved = await loadChat(chats[0].sessionId);
    if (genRef.current !== gen || saved === null) return;
    setMessages(saved);
    setSessionId(chats[0].sessionId);
  }

  async function openChat(id: string) {
    const gen = ++genRef.current;
    const saved = await loadChat(id);
    if (genRef.current !== gen) return;
    // an unloadable transcript must not replace the view with a blank chat;
    // dropping back to the empty state makes the failed bridge visible —
    // and the old session id must not survive: the next message would
    // invisibly resume the conversation the user just left
    if (saved === null) {
      setMessages([]);
      setSessionId(null);
      return;
    }
    setMessages(saved);
    setSessionId(id);
    setShowHistory(false);
  }

  function newChat() {
    genRef.current++;
    setMessages([]);
    setSessionId(null);
    setShowHistory(false);
  }

  async function submit() {
    const text = input.trim();
    if (!text || waiting) return;
    setInput("");
    setMessages((m) => [...m, { role: "user", text }]);
    setWaiting(true);
    const gen = ++genRef.current;
    const out = await sendChat(text, sessionId);
    setWaiting(false);
    if (genRef.current !== gen) return; // conversation switched mid-flight
    if (out.ok) {
      setSessionId(out.sessionId);
      setMessages((m) => [...m, { role: "assistant", text: out.reply }]);
      void listChats().then(setHistory);
    } else {
      setMessages((m) => [...m, {
        role: "assistant",
        text: out.offline
          ? `⚠ ${out.error}`
          : `⚠ The bridge failed: ${out.error}`,
      }]);
    }
  }

  return (
    <div className="flex h-[calc(100vh-8rem)] flex-col">
      <h1 className="text-4xl font-light">Chat</h1>
      <p className="mt-2 font-mono text-xs text-dim">
        YOUR LOCAL CLAUDE SUBSCRIPTION VIA THE CLI BRIDGE — answers are
        grounded in the research snapshot at session start
      </p>

      <div className="mt-4 flex items-center justify-between font-mono text-[11px] tracking-widest">
        <button
          type="button"
          onClick={() => setShowHistory((v) => !v)}
          aria-expanded={showHistory}
          className="text-dim transition-colors hover:text-ink"
        >
          HISTORY ({history === null ? "—" : history.length})
        </button>
        <button
          type="button"
          onClick={newChat}
          className="text-dim transition-colors hover:text-ink"
        >
          NEW CHAT
        </button>
      </div>

      {showHistory && (
        <ul className="mt-2 max-h-48 overflow-y-auto border border-line bg-surface/30">
          {history === null ? (
            <li className="px-4 py-2 font-mono text-[11px] text-dim">
              Can't load chats — is the bridge running?{" "}
              <code className="text-ink">{BRIDGE_START}</code>
            </li>
          ) : history.length === 0 ? (
            <li className="px-4 py-2 font-mono text-[11px] text-dim">
              No saved conversations yet.
            </li>
          ) : (
            history.map((c) => (
              <li key={c.sessionId}>
                <button
                  type="button"
                  onClick={() => { void openChat(c.sessionId); }}
                  className={`flex w-full items-baseline justify-between gap-3 px-4 py-2 text-left text-sm transition-colors hover:bg-surface ${
                    c.sessionId === sessionId ? "text-ink" : "text-dim"}`}
                >
                  <span className="truncate">{c.title}</span>
                  <span className="shrink-0 font-mono text-[10px]">
                    {c.updated?.slice(0, 10) ?? ""}
                  </span>
                </button>
              </li>
            ))
          )}
        </ul>
      )}

      <div className="mt-6 flex min-h-0 flex-1 flex-col overflow-y-auto border border-line bg-surface/30 p-4">
        {messages.length === 0 ? (
          <div className="m-auto max-w-md text-center text-sm text-dim">
            Ask about your research: counts, runs, failures, specs, catalog…
            <div className="mt-3 font-mono text-[11px]">
              requires the local bridge:{" "}
              <code className="text-ink">{BRIDGE_START}</code>
            </div>
          </div>
        ) : (
          messages.map((m, i) =>
            m.role === "assistant" ? (
              <div key={i}
                   className="md my-2 max-w-[85%] self-start border border-line bg-bg px-4 py-3 text-sm leading-relaxed text-ink/95">
                <ReactMarkdown remarkPlugins={[remarkGfm]}>{m.text}</ReactMarkdown>
              </div>
            ) : (
              <div key={i}
                   className="my-2 max-w-[80%] self-end border border-accent/40 bg-accent/15 px-4 py-2.5 text-sm leading-relaxed">
                {m.text}
              </div>
            ),
          )
        )}
        {waiting && (
          <div className="my-2 self-start border border-line bg-bg px-4 py-2.5 font-mono text-xs text-dim">
            thinking…
          </div>
        )}
        <div ref={endRef} />
      </div>

      <form
        className="mt-4 flex gap-3"
        onSubmit={(e) => { e.preventDefault(); void submit(); }}
      >
        <input
          value={input}
          onChange={(e) => setInput(e.target.value)}
          placeholder={waiting ? "waiting for the reply…" : "Ask about your research…"}
          aria-label="Message"
          className="flex-1 border border-line bg-bg px-4 py-3 font-mono text-sm text-ink placeholder:text-dim/60"
        />
        <button
          type="submit"
          disabled={waiting || !input.trim()}
          aria-label="Send message"
          className="flex items-center gap-2 bg-accent px-5 py-3 font-mono text-xs font-medium tracking-widest text-bg transition-opacity disabled:opacity-40"
        >
          <Send className="h-3.5 w-3.5" /> SEND
        </button>
      </form>
    </div>
  );
}
