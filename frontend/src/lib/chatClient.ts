/** Client for the local CLI bridge (local_bridge.py). The bridge shells out
 * to the claude CLI, keeps conversation memory via claude session ids, and
 * saves transcripts under the repo's data/chat/. */

const BRIDGE = import.meta.env.VITE_CHAT_BRIDGE_URL ?? "http://127.0.0.1:8787";

/** How to start the bridge locally; shown wherever the UI needs the user
 * to have it running. Kept in one place so the command can't drift. */
export const BRIDGE_START = "uv run python local_bridge.py";

export type ChatOutcome =
  | { ok: true; reply: string; sessionId: string | null }
  | { ok: false; offline: boolean; error: string };

export interface ChatSummary {
  sessionId: string;
  title: string;
  updated: string | null;
  messageCount: number;
}

export interface ChatTurn {
  role: "user" | "assistant";
  text: string;
}

export async function sendChat(
  message: string,
  sessionId: string | null,
): Promise<ChatOutcome> {
  let res: Response;
  try {
    res = await fetch(`${BRIDGE}/api/chat`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ message, sessionId }),
    });
  } catch {
    return {
      ok: false,
      offline: true,
      error:
        `The local bridge isn't reachable. Start it with: ${BRIDGE_START}. ` +
        "If it is running, your browser may be blocking access to the local network — " +
        "allow the connection when prompted (Chrome asks on first use).",
    };
  }
  const body = (await res.json().catch(() => ({}))) as {
    reply?: string;
    sessionId?: string | null;
    detail?: unknown;
  };
  if (!res.ok) {
    return {
      ok: false,
      offline: false,
      error: errorDetail(body.detail) ?? `bridge returned ${res.status}`,
    };
  }
  if (typeof body.reply !== "string") {
    return { ok: false, offline: false, error: "bridge returned no reply" };
  }
  return { ok: true, reply: body.reply, sessionId: body.sessionId ?? null };
}

/** Saved conversations from the workspace, newest first. null only when
 * the bridge can't be reached (history availability unknown); [] when it
 * answered with none or an unusable payload. */
export async function listChats(): Promise<ChatSummary[] | null> {
  const { reached, body } = await getJson("/api/chats");
  if (!reached) return null;
  if (!Array.isArray(body)) return [];
  return body
    .filter((c): c is Record<string, unknown> =>
      !!c && typeof c === "object" && typeof c.sessionId === "string")
    .map((c) => ({
      sessionId: c.sessionId as string,
      title: typeof c.title === "string" && c.title ? c.title : "Untitled",
      updated: typeof c.updated === "string" ? c.updated : null,
      messageCount: typeof c.messageCount === "number" ? c.messageCount : 0,
    }));
}

/** One saved transcript's turns. null only when the bridge can't be
 * reached; [] when the chat is gone or its transcript is unusable — the
 * bridge answered, there just isn't a conversation behind that id. */
export async function loadChat(sessionId: string): Promise<ChatTurn[] | null> {
  const { reached, body } =
    await getJson(`/api/chats/${encodeURIComponent(sessionId)}`);
  if (!reached) return null;
  const turns = (body as { messages?: unknown })?.messages;
  if (!Array.isArray(turns)) return [];
  return turns.filter(
    (t): t is ChatTurn =>
      !!t && typeof t === "object" &&
      (t.role === "user" || t.role === "assistant") &&
      typeof t.text === "string",
  );
}

/** FastAPI reports failures as `detail`: a string for HTTPException and an
 * array of field errors for a malformed body. */
function errorDetail(detail: unknown): string | null {
  if (typeof detail === "string" && detail.trim()) return detail;
  if (!Array.isArray(detail)) return null;
  const messages = detail
    .map((entry) =>
      entry && typeof entry === "object" &&
      typeof (entry as { msg?: unknown }).msg === "string"
        ? (entry as { msg: string }).msg
        : null)
    .filter((msg): msg is string => !!msg);
  return messages.length > 0 ? messages.join("; ") : null;
}

/** Fetch JSON from the bridge. reached=false only on network failure
 * (bridge not running, or the browser blocking local-network access);
 * a non-2xx status or unparseable body still counts as reached, with
 * body null — the bridge answered, the data just isn't there. */
async function getJson(
  path: string,
): Promise<{ reached: boolean; body: unknown }> {
  try {
    const res = await fetch(`${BRIDGE}${path}`);
    if (!res.ok) return { reached: true, body: null };
    return { reached: true, body: await res.json().catch(() => null) };
  } catch {
    return { reached: false, body: null };
  }
}
