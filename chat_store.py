"""
chat_store.py
-------------
A small, self-contained persistence layer for SkillSync Chatbot
conversations. This is the ONLY new backend module -- it never calls the
model and knows nothing about Groq; it just reads and writes message lists
to disk, exactly the way app.py already shapes them for call_model().

IMPORTANT -- READ BEFORE RELYING ON THIS MODULE
=================================================
The redesign brief (skillsync-frontend-redesign-prompt.md, section 6.2)
asks for chat history to be scoped per logged-in user *if* an auth system
exists in the repo, and to fall back to a simpler shared store otherwise.

app.py, cli.py, groq_client.py, and utils.py were all checked, and none of
them contain any login/authentication code. So this module implements the
explicit fallback design from the brief:

    All chats are stored in ONE shared JSON file (chat_history.json), with
    NO per-user scoping. Every visitor to this running app instance reads
    and writes the same history -- anyone who opens the app can see (and
    delete) every saved chat.

    This is a reasonable default for a small, single-organization tool, but
    it is NOT safe to expose to multiple simultaneous/untrusted users as-is.
    If this app is ever deployed for more than one person at a time, add
    real user-scoping (e.g. a `user_id` column, keyed off whatever identity
    system gets introduced) before that happens.

A JSON file was chosen over the brief's suggested SQLite option because,
without a user_id to key rows by, there's no relational structure this app
actually needs yet -- a single dict-of-chats file is simpler to read, back
up, and reason about. If this ever grows into the multi-user SQLite design
from the brief, only this module needs to change; its public function
signatures (below) are written so that app.py wouldn't need to change at
all.
"""

import json
import os
import threading
from datetime import datetime

DEFAULT_STORE_PATH = "chat_history.json"

# Open question #3 in the brief defaults to "20 most recent, no hard delete
# of older ones unless storage size becomes a concern." Because this store
# is a single unbounded JSON file (unlike a real database), we prune beyond
# this cap on every save so the file doesn't grow forever.
MAX_CHATS_KEPT = 20

# Guards against two Streamlit sessions (e.g. two browser tabs hitting the
# same shared store) reading and writing the file at the same moment.
_lock = threading.Lock()


def init_store(path=DEFAULT_STORE_PATH):
    """Create the store file if it doesn't exist yet. Safe to call every run."""
    if not os.path.exists(path):
        with open(path, "w") as f:
            json.dump({}, f)
    return path


def _load_all(path):
    if not os.path.exists(path):
        return {}
    try:
        with open(path, "r") as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError):
        # A corrupt or unreadable store shouldn't crash the whole app --
        # fail safe with an empty store instead.
        return {}


def _save_all(data, path):
    with open(path, "w") as f:
        json.dump(data, f, indent=2)


def generate_title(first_user_message, max_len=40):
    """Derive a short chat title from the first user message (brief section 6.3)."""
    text = " ".join(first_user_message.split())  # collapse internal whitespace
    if len(text) <= max_len:
        return text or "New chat"
    return text[:max_len].rstrip() + "\u2026"


def save_chat(chat_id, title, mode, messages, path=DEFAULT_STORE_PATH):
    """
    Save (create or overwrite) one chat's full message list.

    `messages` must include the leading system message, in exactly the
    shape call_model() already expects -- this module never reshapes it,
    per the brief's "keep chat_store.py completely separate" instruction.
    """
    with _lock:
        data = _load_all(path)
        now = datetime.now().isoformat()
        existing = data.get(chat_id)
        data[chat_id] = {
            "id": chat_id,
            "title": title or (existing["title"] if existing else "New chat"),
            "mode": mode,
            "messages": messages,
            "created_at": existing["created_at"] if existing else now,
            "updated_at": now,
        }

        if len(data) > MAX_CHATS_KEPT:
            newest_first = sorted(
                data.values(), key=lambda c: c["updated_at"], reverse=True
            )
            keep_ids = {c["id"] for c in newest_first[:MAX_CHATS_KEPT]}
            data = {cid: c for cid, c in data.items() if cid in keep_ids}

        _save_all(data, path)


def load_chat(chat_id, path=DEFAULT_STORE_PATH):
    """Return the message list for one saved chat, or None if it isn't found."""
    chat = _load_all(path).get(chat_id)
    return chat["messages"] if chat else None


def get_chat_meta(chat_id, path=DEFAULT_STORE_PATH):
    """Return {id, title, mode, created_at, updated_at} for one chat (no messages)."""
    chat = _load_all(path).get(chat_id)
    if chat is None:
        return None
    return {k: v for k, v in chat.items() if k != "messages"}


def list_chats(path=DEFAULT_STORE_PATH):
    """Return [{id, title, mode, updated_at}, ...] newest first."""
    data = _load_all(path)
    chats = [
        {
            "id": c["id"],
            "title": c["title"],
            "mode": c["mode"],
            "updated_at": c["updated_at"],
        }
        for c in data.values()
    ]
    chats.sort(key=lambda c: c["updated_at"], reverse=True)
    return chats


def delete_chat(chat_id, path=DEFAULT_STORE_PATH):
    """
    Hard-delete one chat from the store. Open question #2 in the brief
    defaults to hard-delete over soft-delete, so that's what this does.
    Returns True if something was actually deleted.
    """
    with _lock:
        data = _load_all(path)
        if chat_id in data:
            del data[chat_id]
            _save_all(data, path)
            return True
        return False


def delete_all_chats(path=DEFAULT_STORE_PATH):
    """Wipe every saved chat -- backs the sidebar's 'Clear all history' escape hatch."""
    with _lock:
        _save_all({}, path)


def relative_time(iso_timestamp):
    """Turn an ISO timestamp into a short relative label ('2h ago', 'Yesterday', ...)."""
    try:
        then = datetime.fromisoformat(iso_timestamp)
    except (ValueError, TypeError):
        return ""

    delta = datetime.now() - then
    seconds = delta.total_seconds()

    if seconds < 60:
        return "Just now"
    minutes = int(seconds // 60)
    if minutes < 60:
        return f"{minutes}m ago"
    hours = int(minutes // 60)
    if hours < 24:
        return f"{hours}h ago"
    if delta.days == 1:
        return "Yesterday"
    if delta.days < 7:
        return f"{delta.days}d ago"
    return then.strftime("%b %d")
