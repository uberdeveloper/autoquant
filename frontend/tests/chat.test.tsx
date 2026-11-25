import { render, screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import Chat from "../src/ws/screens/Chat";
import { listChats, loadChat, sendChat,
         type ChatSummary, type ChatOutcome, type ChatTurn } from "../src/lib/chatClient";

vi.mock("../src/lib/chatClient", async () => {
  // spread the real module so value exports like BRIDGE_START survive;
  // only the network functions are mocked
  const actual = await vi.importActual<typeof import("../src/lib/chatClient")>(
    "../src/lib/chatClient");
  return {
    ...actual,
    sendChat: vi.fn(),
    listChats: vi.fn(),
    loadChat: vi.fn(),
  };
});

const savedChat = (sessionId: string, title: string) => ({
  sessionId, title, updated: "2026-01-02T10:00:00+00:00", messageCount: 2,
});

async function ask(text: string) {
  await userEvent.type(screen.getByLabelText("Message"), text);
  await userEvent.click(screen.getByLabelText("Send message"));
}

beforeEach(() => {
  vi.clearAllMocks();
  vi.mocked(listChats).mockResolvedValue([]);
  vi.mocked(loadChat).mockResolvedValue([]);
});

describe("Chat screen", () => {
  it("shows the offline note when the bridge fails", async () => {
    vi.mocked(sendChat).mockResolvedValue({
      ok: false, offline: true,
      error: "The local bridge isn't reachable.",
    });
    render(<Chat />);
    await ask("hi");
    await waitFor(() =>
      expect(screen.getByText(/local bridge isn't reachable/)).toBeInTheDocument());
  });

  it("shows the bridge's failure message when the CLI errors", async () => {
    vi.mocked(sendChat).mockResolvedValue({
      ok: false, offline: false,
      error: "claude CLI not found on PATH",
    });
    render(<Chat />);
    await ask("hi");
    await waitFor(() =>
      expect(screen.getByText(/claude CLI not found on PATH/)).toBeInTheDocument());
  });

  it("renders the reply as an assistant bubble", async () => {
    vi.mocked(sendChat).mockResolvedValue({
      ok: true, reply: "7 backtests", sessionId: "s1",
    });
    render(<Chat />);
    await ask("how many runs?");
    await waitFor(() => expect(screen.getByText("7 backtests")).toBeInTheDocument());
  });

  it("renders markdown in the reply — bold, lists, tables", async () => {
    vi.mocked(sendChat).mockResolvedValue({
      ok: true,
      reply: "Best is **spy-200dma**:\n\n- sharpe **1.06**\n\n| run | state |\n|---|---|\n| a | error |",
      sessionId: "s1",
    });
    render(<Chat />);
    await ask("best run?");
    await waitFor(() => {
      const strongs = [...document.querySelectorAll(".md strong")].map(s => s.textContent);
      expect(strongs).toContain("1.06");
      expect(document.querySelector(".md li")?.textContent).toContain("1.06");
      expect(document.querySelector(".md table")).not.toBeNull();
    });
  });
});

describe("Chat history", () => {
  it("restores the most recent saved conversation on load", async () => {
    vi.mocked(listChats).mockResolvedValue([
      savedChat("b", "sharpe leaderboard?"),
      savedChat("a", "how many runs?"),
    ]);
    vi.mocked(loadChat).mockResolvedValue([
      { role: "user", text: "how many runs?" },
      { role: "assistant", text: "7 backtests" },
    ]);
    render(<Chat />);
    await waitFor(() =>
      expect(screen.getByText("7 backtests")).toBeInTheDocument());
    expect(screen.getByText("how many runs?")).toBeInTheDocument();
    expect(loadChat).toHaveBeenCalledWith("b");
  });

  it("continues the restored conversation instead of starting over", async () => {
    vi.mocked(listChats).mockResolvedValue([savedChat("b", "runs?")]);
    vi.mocked(loadChat).mockResolvedValue([
      { role: "user", text: "runs?" },
      { role: "assistant", text: "7" },
    ]);
    vi.mocked(sendChat).mockResolvedValue({
      ok: true, reply: "8", sessionId: "b",
    });
    render(<Chat />);
    await waitFor(() => expect(screen.getByText("7")).toBeInTheDocument());
    await ask("and now?");
    await waitFor(() => expect(sendChat).toHaveBeenCalledWith("and now?", "b"));
  });

  it("lists saved conversations and opens one", async () => {
    vi.mocked(listChats).mockResolvedValue([
      savedChat("b", "newest question"),
      savedChat("a", "older question"),
    ]);
    vi.mocked(loadChat).mockImplementation(async (id) =>
      id === "a"
        ? [{ role: "user", text: "older question" },
           { role: "assistant", text: "older answer" }]
        : [{ role: "user", text: "newest question" },
           { role: "assistant", text: "newest answer" }]);
    render(<Chat />);
    await waitFor(() =>
      expect(screen.getByText("newest answer")).toBeInTheDocument());
    await userEvent.click(screen.getByRole("button", { name: /HISTORY \(2\)/ }));
    await userEvent.click(screen.getByText("older question"));
    await waitFor(() =>
      expect(screen.getByText("older answer")).toBeInTheDocument());
    expect(screen.queryByText("newest answer")).not.toBeInTheDocument();
  });

  it("starts a fresh conversation after NEW CHAT", async () => {
    vi.mocked(listChats).mockResolvedValue([savedChat("b", "runs?")]);
    vi.mocked(loadChat).mockResolvedValue([
      { role: "user", text: "runs?" },
      { role: "assistant", text: "7" },
    ]);
    vi.mocked(sendChat).mockResolvedValue({
      ok: true, reply: "fresh", sessionId: "c",
    });
    render(<Chat />);
    await waitFor(() => expect(screen.getByText("7")).toBeInTheDocument());
    await userEvent.click(screen.getByRole("button", { name: "NEW CHAT" }));
    expect(screen.queryByText("7")).not.toBeInTheDocument();
    await ask("first question");
    await waitFor(() =>
      expect(sendChat).toHaveBeenCalledWith("first question", null));
  });

  it("refreshes the saved list after an answer", async () => {
    vi.mocked(sendChat).mockResolvedValue({
      ok: true, reply: "PONG", sessionId: "s1",
    });
    render(<Chat />);
    await waitFor(() => expect(listChats).toHaveBeenCalledTimes(1));
    await ask("hi");
    await waitFor(() => expect(listChats).toHaveBeenCalledTimes(2));
  });

  it("says so when nothing has been saved yet", async () => {
    render(<Chat />);
    await waitFor(() => expect(listChats).toHaveBeenCalled());
    await userEvent.click(screen.getByRole("button", { name: /HISTORY \(0\)/ }));
    expect(screen.getByText(/No saved conversations yet/)).toBeInTheDocument();
  });

  it("drops a reply that lands after the user started a new chat", async () => {
    let resolveSend!: (v: ChatOutcome) => void;
    vi.mocked(sendChat).mockImplementation(
      () => new Promise((resolve) => { resolveSend = resolve; }));
    render(<Chat />);
    await ask("hi");
    await userEvent.click(screen.getByRole("button", { name: "NEW CHAT" }));
    resolveSend({ ok: true, reply: "stale reply", sessionId: "old" });
    // the send is no longer pending…
    await waitFor(() => expect(
      screen.getByPlaceholderText("Ask about your research…")).toBeInTheDocument());
    // …but the stale reply never enters the fresh conversation
    expect(screen.queryByText("stale reply")).not.toBeInTheDocument();
  });

  it("keeps the fresh chat when the initial restore resolves late", async () => {
    let resolveList!: (v: ChatSummary[]) => void;
    vi.mocked(listChats).mockImplementation(
      () => new Promise((resolve) => { resolveList = resolve; }));
    render(<Chat />);
    await userEvent.click(screen.getByRole("button", { name: "NEW CHAT" }));
    resolveList([savedChat("old", "Old chat")]);
    await waitFor(() => expect(
      screen.getByText(/Ask about your research/)).toBeInTheDocument());
    expect(loadChat).not.toHaveBeenCalled();
  });

  it("keeps the fresh chat when a history open resolves late", async () => {
    let resolveLoad!: (v: ChatTurn[]) => void;
    vi.mocked(loadChat).mockImplementation(
      () => new Promise((resolve) => { resolveLoad = resolve; }));
    vi.mocked(listChats).mockResolvedValue([savedChat("old", "Old chat")]);
    render(<Chat />);
    await waitFor(() => expect(
      screen.getByRole("button", { name: /HISTORY \(1\)/ })).toBeInTheDocument());
    await userEvent.click(screen.getByRole("button", { name: /HISTORY \(1\)/ }));
    await userEvent.click(screen.getByText("Old chat"));
    await userEvent.click(screen.getByRole("button", { name: "NEW CHAT" }));
    resolveLoad([{ role: "user", text: "old question" }]);
    await waitFor(() => expect(
      screen.getByText(/Ask about your research/)).toBeInTheDocument());
    expect(screen.queryByText("old question")).not.toBeInTheDocument();
  });

  it("marks history as unavailable when the bridge can't be reached", async () => {
    vi.mocked(listChats).mockResolvedValue(null);
    render(<Chat />);
    const toggle = await screen.findByRole(
      "button", { name: /HISTORY \(—\)/ });
    await userEvent.click(toggle);
    expect(screen.getByText(/Can't load chats/)).toBeInTheDocument();
    expect(screen.queryByText(/No saved conversations yet/))
      .not.toBeInTheDocument();
  });

  it("falls back to the empty-state hint when a saved chat can't be loaded", async () => {
    vi.mocked(listChats).mockResolvedValue([
      savedChat("b", "newest question"),
      savedChat("a", "older question"),
    ]);
    vi.mocked(loadChat).mockResolvedValueOnce([
      { role: "user", text: "newest question" },
      { role: "assistant", text: "newest answer" },
    ]);
    render(<Chat />);
    await waitFor(() =>
      expect(screen.getByText("newest answer")).toBeInTheDocument());
    vi.mocked(loadChat).mockResolvedValue(null);
    await userEvent.click(screen.getByRole("button", { name: /HISTORY \(2\)/ }));
    await userEvent.click(screen.getByText("older question"));
    // an unloadable transcript must not masquerade as a blank conversation
    await waitFor(() => expect(
      screen.queryByText("newest answer")).not.toBeInTheDocument());
    expect(screen.getByText(/Ask about your research/)).toBeInTheDocument();
  });

  it("sends the next message fresh after a saved chat fails to load", async () => {
    vi.mocked(listChats).mockResolvedValue([
      savedChat("b", "newest question"),
      savedChat("a", "older question"),
    ]);
    vi.mocked(loadChat).mockResolvedValueOnce([
      { role: "user", text: "newest question" },
      { role: "assistant", text: "newest answer" },
    ]);
    vi.mocked(sendChat).mockResolvedValue({
      ok: true, reply: "PONG", sessionId: "c",
    });
    render(<Chat />);
    await waitFor(() =>
      expect(screen.getByText("newest answer")).toBeInTheDocument());
    vi.mocked(loadChat).mockResolvedValue(null);
    await userEvent.click(screen.getByRole("button", { name: /HISTORY \(2\)/ }));
    await userEvent.click(screen.getByText("older question"));
    await waitFor(() => expect(
      screen.getByText(/Ask about your research/)).toBeInTheDocument());
    // the failed switch must not leave the previous conversation's session
    // id live: the next message would invisibly resume and extend it
    await ask("next question");
    await waitFor(() =>
      expect(sendChat).toHaveBeenCalledWith("next question", null));
  });

  it("still fills the history list when NEW CHAT beats the initial restore", async () => {
    let resolveList!: (v: ChatSummary[] | null) => void;
    vi.mocked(listChats).mockImplementation(
      () => new Promise((resolve) => { resolveList = resolve; }));
    render(<Chat />);
    await userEvent.click(screen.getByRole("button", { name: "NEW CHAT" }));
    resolveList([savedChat("old", "Old chat")]);
    // the history list is global: a restore that loses the conversation
    // race must still populate it, not leave a false HISTORY (0)
    await waitFor(() =>
      expect(screen.getByRole("button", { name: /HISTORY \(1\)/ }))
        .toBeInTheDocument());
    expect(loadChat).not.toHaveBeenCalled();
  });
});
