#!/usr/bin/env python3
"""LOCAL BRIDGE — connects the dashboard's Chat tab to the local claude CLI.

The claude CLI does everything (model, auth via the user's subscription,
conversation memory via --resume); this server only formats and saves
transcripts under the repo's data/chat/. Runs on 127.0.0.1 by default —
not meant for public exposure.

    uv run python local_bridge.py [--port 8787]

Frontend: set VITE_CHAT_BRIDGE_URL (default http://127.0.0.1:8787).
"""
from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import re
import signal
import tempfile
import threading
import uuid
from datetime import datetime, timezone
from pathlib import Path

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

ROOT = Path(__file__).resolve().parent
DIGEST_PATH = ROOT / "frontend" / "src" / "content" / "snapshot.json"
CHAT_DIR = ROOT / "data" / "chat"
TIMEOUT_S = 300
TITLE_MAX = 60
MAX_MESSAGE_CHARS = 16_000
MAX_BODY_BYTES = 64 * 1024
REPLAY_MAX_TURNS = 20
REPLAY_MAX_CHARS = 8_000
SESSION_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_-]{0,63}$")
LOOPBACK_HOSTS = {"127.0.0.1", "localhost", "::1"}
log = logging.getLogger("local_bridge")

# Per-session save locks: save_turn is read-modify-write, so two concurrent
# POSTs for the same session would silently lose one exchange.
_save_locks: dict[str, threading.Lock] = {}
_save_locks_guard = threading.Lock()


def _save_lock(session_id: str) -> threading.Lock:
    with _save_locks_guard:
        lock = _save_locks.get(session_id)
        if lock is None:
            lock = _save_locks[session_id] = threading.Lock()
        return lock

app = FastAPI(title="autoquant local bridge")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://autoquant-blush.vercel.app"],
    # any local dev port (5173, 4173, 4199, …); non-local origins stay blocked
    allow_origin_regex=r"https?://(localhost|127\.0\.0\.1)(:\d+)?",
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
    # Chrome sends a Private Network Access preflight when the deployed
    # HTTPS app fetches this loopback server; without this it blocks every
    # request (and Safari/Firefox ignore it). Still gated by the origin
    # allowlist above.
    allow_private_network=True,
)


@app.middleware("http")
async def limit_body(request: Request, call_next):
    """Guard the bridge before FastAPI parses anything.

    Size cap, then loopback-only access (a browser cannot be trusted to
    enforce CORS on its own: a text/plain cross-site POST would still run,
    and DNS rebinding would make the bridge same-origin), then require
    application/json on POSTs — a browser cannot send that content type
    cross-origin without a preflight, which the CORS policy rejects.
    """
    raw = request.headers.get("content-length")
    if raw is None and request.method == "POST":
        return JSONResponse({"detail": "content-length required"},
                            status_code=411)
    if raw is not None:
        try:
            too_large = int(raw) > MAX_BODY_BYTES
        except ValueError:
            return JSONResponse({"detail": "invalid content-length"},
                                status_code=400)
        if too_large:
            return JSONResponse({"detail": "request body too large"},
                                status_code=413)
    host = request.headers.get("host") or ""
    candidates = {host.strip("[]"), host.rsplit(":", 1)[0].strip("[]")}
    if host and not candidates & LOOPBACK_HOSTS:
        return JSONResponse({"detail": "loopback access only"},
                            status_code=403)
    if request.method == "POST":
        ctype = request.headers.get("content-type", "").split(";")[0].strip()
        if ctype != "application/json":
            return JSONResponse({"detail": "use application/json"},
                                status_code=415)
    return await call_next(request)


class CLIError(Exception):
    """The claude CLI failed (nonzero exit / is_error / unparseable output)."""


def build_claude_args(message: str, session_id: str | None) -> list[str]:
    """claude CLI argv: --resume continues a session's full memory."""
    args = []
    if session_id:
        args += ["--resume", session_id]
    args += ["-p", message, "--output-format", "json"]
    return args


def parse_reply(stdout: str) -> dict:
    """Extract {reply, sessionId} from the CLI's JSON output."""
    try:
        data = json.loads(stdout)
    except json.JSONDecodeError as exc:
        raise CLIError(f"unparseable CLI output: {exc}") from exc
    if data.get("is_error"):
        raise CLIError(str(data.get("result") or "CLI reported an error"))
    reply = data.get("result")
    if not isinstance(reply, str) or not reply.strip():
        raise CLIError("CLI returned an empty reply")
    return {"reply": reply, "sessionId": data.get("session_id")}


async def run_claude(message: str, session_id: str | None) -> dict:
    """Spawn the claude CLI and parse its JSON output."""
    try:
        proc = await asyncio.create_subprocess_exec(
            "claude", *build_claude_args(message, session_id),
            stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE,
            start_new_session=True,
        )
    except FileNotFoundError as exc:
        raise CLIError("claude CLI not found on PATH") from exc
    try:
        stdout, stderr = await asyncio.wait_for(proc.communicate(), TIMEOUT_S)
    except asyncio.TimeoutError:
        try:
            os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
        except (ProcessLookupError, PermissionError):
            try:
                proc.kill()
            except ProcessLookupError:
                pass
        await proc.wait()
        raise CLIError(f"claude CLI timed out after {TIMEOUT_S}s") from None
    if proc.returncode != 0:
        raise CLIError((stderr or b"").decode(errors="ignore")[-300:]
                       or f"claude CLI exited {proc.returncode}")
    return parse_reply(stdout.decode(errors="ignore"))


def seed_message(message: str, digest: str, history: str = "") -> str:
    """Ground the assistant in the digest, replaying earlier turns if any."""
    earlier = (f"--- EARLIER TURNS ---\n{history}\n"
               "--- END EARLIER TURNS ---\n\n" if history else "")
    return (
        "You are AutoQuant's research assistant. Answer only from the "
        "research snapshot below — if something is not in it, say so; "
        "never invent results, metrics, or citations.\n\n"
        f"--- RESEARCH SNAPSHOT ---\n{digest}\n--- END SNAPSHOT ---\n\n"
        f"{earlier}{message}"
    )


def load_digest() -> str:
    """Compact research digest; grounding is optional, never load-bearing."""
    try:
        return _digest_text()
    except (OSError, json.JSONDecodeError, TypeError, AttributeError) as exc:
        log.warning("could not load digest: %s", exc)
        return ("Research snapshot unavailable. Answer from the question "
                "alone and say you could not load the research snapshot.")


def _digest_text() -> str:
    """Compact research digest from the build-time snapshot."""
    snapshot = json.loads(DIGEST_PATH.read_text())
    counts = snapshot.get("counts", {})
    workspace_path = DIGEST_PATH.with_name("workspace.json")
    try:
        w = json.loads(workspace_path.read_text())
    except (OSError, json.JSONDecodeError):
        w = snapshot.get("workspace", {})
    runs = sorted(w.get("runs", []), key=lambda r: r.get("date") or "",
                  reverse=True)
    latest = "; ".join(
        f"{r.get('slug')} [{r.get('status')} sharpe={r.get('netSharpe')}]"
        for r in runs[:5])
    failed = "; ".join(str(f.get("slug", "")) for f in w.get("failed", [])[:5]
                       if isinstance(f, dict)) or "none"
    specs = "; ".join(s.get("slug", "") for s in w.get("specs", [])[:8])
    return (
        f"Generated {snapshot.get('generatedAt')}. "
        f"Counts: {counts}. "
        f"Latest runs: {latest}. "
        f"Currently failing codegen: {failed}. "
        f"Specs include: {specs}. "
        f"Catalog: {w.get('catalogCount', 0)} data series."
    )


def resumable(session_id: str | None) -> bool:
    """claude --resume only accepts the UUID ids its own sessions carry."""
    try:
        uuid.UUID(session_id or "")
    except ValueError:
        return False
    return True


def history_text(session_id: str | None) -> str:
    """Saved turns for a transcript the CLI cannot resume, newest last."""
    if not session_id:
        return ""
    chat = load_chat(session_id) or {"messages": []}
    lines, used = [], 0
    for turn in reversed(chat["messages"][-REPLAY_MAX_TURNS:]):
        text = str(turn.get("text", ""))[:2_000]
        if used + len(text) > REPLAY_MAX_CHARS:
            break
        used += len(text)
        lines.append(f"{turn.get('role', 'user').title()}: {text}")
    return "\n\n".join(reversed(lines))


class ChatBody(BaseModel):
    message: str = Field(min_length=1, max_length=MAX_MESSAGE_CHARS)
    sessionId: str | None = None


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _chat_path(session_id: str) -> Path:
    """Path for a transcript; ids are validated so they stay inside CHAT_DIR."""
    if not SESSION_ID.match(session_id or ""):
        raise ValueError(f"invalid session id: {session_id!r}")
    return CHAT_DIR / f"{session_id}.json"


def _title(question: str) -> str:
    return " ".join(question.split())[:TITLE_MAX]


def load_chat(session_id: str) -> dict | None:
    """A saved transcript, or None when it is missing or unreadable."""
    try:
        data = json.loads(_chat_path(session_id).read_text())
    except (OSError, json.JSONDecodeError):
        return None
    if not isinstance(data, dict) or not isinstance(data.get("messages"), list):
        return None
    return data


def save_turn(session_id: str, role: str, text: str) -> None:
    """Append one turn to a transcript stored inside the repo workspace."""
    path = _chat_path(session_id)
    chat = load_chat(session_id) or {
        "sessionId": session_id, "title": "", "created": _now(),
        "messages": [],
    }
    chat["messages"].append({"role": role, "text": text, "at": _now()})
    if not chat.get("title") and role == "user":
        chat["title"] = _title(text)
    chat["updated"] = _now()
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    # Replace atomically so a crash cannot leave a half-written transcript
    with tempfile.NamedTemporaryFile(
            "w", dir=path.parent, delete=False, suffix=".tmp") as tmp:
        json.dump(chat, tmp)
        tmp_path = Path(tmp.name)
    os.chmod(tmp_path, 0o600)
    os.replace(tmp_path, path)


def list_chats() -> list[dict]:
    """Saved conversations, most recently active first."""
    if not CHAT_DIR.is_dir():
        return []
    chats = []
    for path in CHAT_DIR.glob("*.json"):
        try:
            chat = load_chat(path.stem)
        except ValueError:
            continue
        if chat is None:
            continue
        chats.append({
            "sessionId": chat.get("sessionId") or path.stem,
            "title": chat.get("title") or path.stem,
            "updated": chat.get("updated"),
            "messageCount": len(chat["messages"]),
        })
    chats.sort(key=lambda c: c.get("updated") or "", reverse=True)
    return chats


def record_turn(session_id: str | None, question: str, reply: str) -> None:
    """Save the exchange; a storage problem must never lose the reply."""
    if not session_id:
        return
    try:
        save_turn(session_id, "user", question)
        save_turn(session_id, "assistant", reply)
    except (OSError, ValueError) as exc:
        log.warning("could not save chat history: %s", exc)


@app.get("/api/health")
def health() -> dict:
    return {"ok": True}


@app.get("/api/chats")
def chats() -> list[dict]:
    return list_chats()


@app.get("/api/chats/{session_id}")
def chat_transcript(session_id: str) -> dict:
    try:
        chat = load_chat(session_id)
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc
    if chat is None:
        raise HTTPException(404, "no such chat")
    return chat


@app.post("/api/chat")
async def chat(body: ChatBody) -> dict:
    if not body.message.strip():
        raise HTTPException(400, "message must not be empty")
    if body.sessionId is not None and not SESSION_ID.match(body.sessionId):
        raise HTTPException(400, "invalid session id")
    resume = body.sessionId if resumable(body.sessionId) else None
    if body.sessionId is None:
        message = seed_message(body.message, load_digest())
    elif resume is None:
        # saved under an id --resume rejects: replay the turns instead
        message = seed_message(body.message, load_digest(),
                               history_text(body.sessionId))
    else:
        message = body.message
    try:
        out = await run_claude(message, resume)
    except CLIError as exc:
        if resume is None or "timed out" in str(exc):
            raise HTTPException(502, str(exc)) from exc
        # A structurally valid UUID the CLI no longer knows must not brick
        # the thread: replay the saved transcript into a fresh session.
        message = seed_message(body.message, load_digest(),
                               history_text(body.sessionId))
        try:
            out = await run_claude(message, None)
        except CLIError as retry_exc:
            raise HTTPException(502, str(retry_exc)) from retry_exc
        out["sessionId"] = body.sessionId
    if resume is None and body.sessionId is not None:
        out["sessionId"] = body.sessionId
    with _save_lock(out["sessionId"] or ""):
        record_turn(out["sessionId"], body.message.strip(), out["reply"])
    return out


def main_with(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8787)
    ap.add_argument("--host", default="127.0.0.1")
    args = ap.parse_args(argv)
    import uvicorn
    uvicorn.run(app, host=args.host, port=args.port, log_level="warning")
    return 0


if __name__ == "__main__":
    raise SystemExit(main_with())
