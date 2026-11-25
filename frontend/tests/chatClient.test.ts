import { listChats, loadChat, sendChat } from "../src/lib/chatClient";

describe("sendChat", () => {
  afterEach(() => vi.unstubAllGlobals());

  it("posts to the bridge and unwraps the reply", async () => {
    const fetchMock = vi.fn().mockResolvedValue({
      ok: true,
      json: () => Promise.resolve({ reply: "PONG", sessionId: "abc" }),
    });
    vi.stubGlobal("fetch", fetchMock);
    const out = await sendChat("hi", null);
    expect(out).toEqual({ ok: true, reply: "PONG", sessionId: "abc" });
    expect(fetchMock.mock.calls[0][0]).toBe("http://127.0.0.1:8787/api/chat");
    expect(JSON.parse(fetchMock.mock.calls[0][1].body)).toEqual({
      message: "hi", sessionId: null,
    });
  });

  it("flags offline when the bridge is unreachable", async () => {
    vi.stubGlobal("fetch", vi.fn().mockRejectedValue(new TypeError("fail")));
    const out = await sendChat("hi", null);
    expect(out).toMatchObject({ ok: false, offline: true });
    // a running bridge can still be blocked by the browser's local-network
    // gate, so the message must cover that case too
    expect((out as { ok: false; error: string }).error)
      .toMatch(/local network/i);
  });

  it("surfaces the bridge's detail without the offline flag", async () => {
    vi.stubGlobal("fetch", vi.fn().mockResolvedValue({
      ok: false,
      status: 502,
      json: () => Promise.resolve({ detail: "boom from cli" }),
    }));
    const out = await sendChat("hi", "abc");
    expect(out).toMatchObject({ ok: false, offline: false, error: "boom from cli" });
  });

  it("surfaces a body-validation failure as its messages", async () => {
    vi.stubGlobal("fetch", vi.fn().mockResolvedValue({
      ok: false,
      status: 422,
      json: () => Promise.resolve({
        detail: [{ msg: "String should have at least 1 character" }],
      }),
    }));
    const out = await sendChat("", null);
    expect(out).toMatchObject({
      ok: false,
      offline: false,
      error: "String should have at least 1 character",
    });
  });

  it("falls back to the status when a failure has no usable detail", async () => {
    vi.stubGlobal("fetch", vi.fn().mockResolvedValue({
      ok: false,
      status: 413,
      json: () => Promise.resolve({ detail: [null, "junk"] }),
    }));
    const out = await sendChat("hi", "abc");
    expect(out).toMatchObject({ ok: false, offline: false, error: "bridge returned 413" });
  });
});

describe("listChats", () => {
  afterEach(() => vi.unstubAllGlobals());

  const reply = (body: unknown) =>
    vi.fn().mockResolvedValue({ ok: true, json: () => Promise.resolve(body) });

  it("returns saved conversations with missing fields defaulted", async () => {
    vi.stubGlobal("fetch", reply([
      { sessionId: "a", title: "runs?", updated: "2026-01-02", messageCount: 4 },
      { sessionId: "b" },
    ]));
    expect(await listChats()).toEqual([
      { sessionId: "a", title: "runs?", updated: "2026-01-02", messageCount: 4 },
      { sessionId: "b", title: "Untitled", updated: null, messageCount: 0 },
    ]);
  });

  it("drops entries without a session id", async () => {
    vi.stubGlobal("fetch", reply([
      null, "junk", { title: "no id" }, { sessionId: 7 }, { sessionId: "ok" },
    ]));
    expect(await listChats()).toEqual([
      { sessionId: "ok", title: "Untitled", updated: null, messageCount: 0 },
    ]);
  });

  it("returns nothing when the payload is not a list", async () => {
    vi.stubGlobal("fetch", reply({ chats: [] }));
    expect(await listChats()).toEqual([]);
  });

  it("returns null when the bridge is offline", async () => {
    vi.stubGlobal("fetch", vi.fn().mockRejectedValue(new TypeError("fail")));
    expect(await listChats()).toEqual(null);
  });
});

describe("loadChat", () => {
  afterEach(() => vi.unstubAllGlobals());

  it("returns the transcript turns", async () => {
    const fetchMock = vi.fn().mockResolvedValue({
      ok: true,
      json: () => Promise.resolve({
        sessionId: "a",
        messages: [
          { role: "user", text: "hi" },
          { role: "assistant", text: "PONG" },
        ],
      }),
    });
    vi.stubGlobal("fetch", fetchMock);
    expect(await loadChat("a")).toEqual([
      { role: "user", text: "hi" },
      { role: "assistant", text: "PONG" },
    ]);
    expect(fetchMock.mock.calls[0][0]).toBe(
      "http://127.0.0.1:8787/api/chats/a");
  });

  it("drops malformed turns", async () => {
    vi.stubGlobal("fetch", vi.fn().mockResolvedValue({
      ok: true,
      json: () => Promise.resolve({
        messages: [
          null, { role: "system", text: "x" }, { role: "user" },
          { role: "user", text: 5 }, { role: "user", text: "kept" },
        ],
      }),
    }));
    expect(await loadChat("a")).toEqual([{ role: "user", text: "kept" }]);
  });

  it("returns an empty transcript for a chat the bridge no longer has", async () => {
    vi.stubGlobal("fetch", vi.fn().mockResolvedValue({ ok: false, status: 404 }));
    expect(await loadChat("gone")).toEqual([]);
  });

  it("returns null when the bridge is offline", async () => {
    vi.stubGlobal("fetch", vi.fn().mockRejectedValue(new TypeError("fail")));
    expect(await loadChat("gone")).toEqual(null);
  });
});
