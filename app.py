"""
app.py
------
The web-based frontend for the "SkillSync Chatbot", built with Streamlit.

This is a visual and UX redesign of the original app.py, per
skillsync-frontend-redesign-prompt.md. It still does NOT reinvent any of
the logic in cli.py, groq_client.py, or utils.py -- call_model(),
load_client(), and get_system_prompt() are used exactly as before, with
the same signatures. Everything new here is either presentation (colors,
layout, avatars, the intro animation) or the new chat_store.py-backed
history feature described in section 6 of the brief.

Decisions made while implementing (flagged explicitly, as the brief asks
for in its "open questions" section, rather than assumed silently):

  1. No login/auth file exists anywhere in this codebase (app.py, cli.py,
     groq_client.py, utils.py were all checked). Chat history therefore
     uses the single-shared-store fallback from brief section 6.2 -- see
     chat_store.py's module docstring for exactly what that means and
     what its limitation is.
  2. Streamlit has no supported way to pre-fill st.chat_input's text from
     Python, so the suggested-prompt chips (brief section 5.2) send that
     prompt immediately when clicked, rather than "populating" the input
     box for the user to press Enter on. This is the common pattern other
     Streamlit chat UIs use for the same reason.
  3. Deleted chats are hard-deleted (brief's stated default, open question
     #2), and the sidebar keeps the 20 most recent chats (open question #3
     default), pruning older ones on save.
  4. Switching the Q&A / Summarizer mode mid-conversation starts a fresh
     chat (matching the original app's behavior), since the two modes use
     different system prompts and mixing them in one history would be
     incoherent context for the model.

HOW TO RUN THIS FILE:
    python -m streamlit run app.py

FIRST TIME ONLY -- install streamlit before running:
    python -m pip install streamlit
"""

import json
import os
import uuid
from datetime import datetime

import streamlit as st
from streamlit.components.v1 import html as components_html

from groq_client import load_client, call_model
from utils import get_system_prompt
from chat_store import (
    init_store,
    save_chat,
    load_chat,
    get_chat_meta,
    list_chats,
    delete_chat,
    delete_all_chats,
    generate_title,
    relative_time,
)

APP_DIR = os.path.dirname(os.path.abspath(__file__))

SUGGESTED_PROMPTS = [
    "What is SkillSync?",
    "Tell me about skillIT",
    "What courses do you offer?",
    "How do I get started?",
]

# A single small four-point "spark" mark used as the app's wordmark, in place
# of a generic emoji brain. Uses currentColor so the same markup works both
# on the white header banner and against the sidebar/page text color.
SPARK_ICON_SVG = """
<svg width="{size}" height="{size}" viewBox="0 0 24 24" fill="none"
     xmlns="http://www.w3.org/2000/svg" style="vertical-align:-6px; flex-shrink:0;">
  <path d="M12 2.5L14.2 9.3L21 12L14.2 14.7L12 21.5L9.8 14.7L3 12L9.8 9.3L12 2.5Z"
        fill="currentColor"/>
</svg>
"""


def _spark_icon(size=24):
    return SPARK_ICON_SVG.format(size=size)


def _copy_button_html(text, key):
    """A tiny client-side (no server round-trip) copy-to-clipboard control
    rendered under an assistant reply. Pure HTML/JS so it works safely
    inside st.markdown(unsafe_allow_html=True) without a custom component."""
    safe_text = json.dumps(text)
    return f"""
    <div class="copy-btn-row">
      <button type="button" class="copy-btn" id="copy-{key}"
        onclick="navigator.clipboard.writeText({safe_text});
                 this.innerText='Copied';
                 this.classList.add('copied');
                 setTimeout(() => {{ this.innerText='Copy'; this.classList.remove('copied'); }}, 1200);">
        Copy
      </button>
    </div>
    """

# ============================================================
# PAGE SETUP
# ============================================================
st.set_page_config(
    page_title="SkillSync Assistant",
    page_icon="\U0001F9E0",
    layout="centered",
    # Left at Streamlit's default so its own mobile sidebar-collapse
    # behavior is never overridden -- see redesign-prompt.md section 5.4.
    initial_sidebar_state="auto",
)

init_store()

# ============================================================
# THEME DETECTION
# ============================================================
# Match whichever theme Streamlit itself resolved (the user's own choice
# in Settings, which defaults to "Use system setting") rather than
# hand-rolling a separate light/dark toggle that would fight it.
# st.context.theme is fairly recent, so this degrades gracefully on older
# Streamlit versions instead of crashing the app.
try:
    THEME_BASE = st.context.theme.base
except Exception:
    THEME_BASE = "light"

IS_DARK = THEME_BASE == "dark"

# ============================================================
# CUSTOM STYLING (CSS)
# ============================================================
if IS_DARK:
    THEME_VARS = """
        --bg: #14101F;
        --bg-elevated: #1F1A33;
        --text: #F5F3FF;
        --text-muted: #B8AED9;
        --border: #332A4D;
        --user-bubble-bg: rgba(167, 139, 250, 0.20);
        --user-bubble-border: rgba(167, 139, 250, 0.35);
        --assistant-bubble-bg: #1B1530;
        --chip-bg: #1F1A33;
        --chip-border: #3A3159;
    """
else:
    THEME_VARS = """
        --bg: #FFFFFF;
        --bg-elevated: #FAFAFC;
        --text: #1F2333;
        --text-muted: #6B6580;
        --border: #E5E1F5;
        --user-bubble-bg: rgba(124, 58, 237, 0.09);
        --user-bubble-border: rgba(124, 58, 237, 0.22);
        --assistant-bubble-bg: #FAFAFC;
        --chip-bg: #FFFFFF;
        --chip-border: #E5E1F5;
    """

st.markdown(
    f"""
    <style>
    :root {{
        --accent-start: #4F46E5;
        --accent-mid: #7C3AED;
        --accent-end: #DB2777;
        --gradient: linear-gradient(90deg, var(--accent-start) 0%, var(--accent-mid) 50%, var(--accent-end) 100%);
        {THEME_VARS}
    }}

    html, body, [class*="css"] {{
        font-family: -apple-system, "Segoe UI", "Inter", sans-serif;
        font-size: 16.5px;
    }}

    /* ---------- Header banner: the single biggest use of the gradient ---------- */
    .header-banner {{
        background: var(--gradient);
        padding: 32px 24px;
        border-radius: 16px;
        margin-bottom: 28px;
        text-align: center;
    }}
    .header-banner h1 {{
        color: white;
        font-size: 1.9rem;
        font-weight: 700;
        margin: 0;
    }}
    .header-banner p {{
        color: #F3EEFF;
        margin-top: 8px;
        font-size: 0.98rem;
    }}

    /* ---------- Landing / empty state ---------- */
    .landing-hero {{
        text-align: center;
        padding: 8px 8px 4px 8px;
    }}
    .landing-hero h2 {{
        font-size: 1.5rem;
        font-weight: 700;
        color: var(--text);
        margin-bottom: 6px;
    }}
    .landing-hero p {{
        color: var(--text-muted);
        font-size: 1rem;
        max-width: 480px;
        margin: 0 auto 22px auto;
        line-height: 1.5;
    }}

    /* Suggested-prompt chip buttons. Streamlit has renamed this button's
       data-testid across versions (kind="secondary" -> baseButton-secondary
       -> stBaseButton-secondary), so all three are targeted for safety. */
    button[kind="secondary"],
    button[data-testid="baseButton-secondary"],
    button[data-testid="stBaseButton-secondary"] {{
        border-radius: 999px;
        border: 1px solid var(--chip-border);
        background: var(--chip-bg);
        color: var(--text);
        min-height: 44px;
        font-size: 0.9rem;
        transition: border-color 150ms ease, transform 150ms ease;
    }}
    button[kind="secondary"]:hover,
    button[data-testid="baseButton-secondary"]:hover,
    button[data-testid="stBaseButton-secondary"]:hover {{
        border-color: var(--accent-mid);
        color: var(--accent-mid);
        transform: translateY(-1px);
    }}

    /* Primary buttons (New chat, active history item, etc.) carry the gradient */
    button[kind="primary"],
    button[data-testid="baseButton-primary"],
    button[data-testid="stBaseButton-primary"] {{
        background: var(--gradient) !important;
        border: none !important;
        min-height: 44px;
        font-weight: 600;
    }}

    /* ---------- Chat bubbles: user vs. assistant must read as distinct ---------- */
    div[data-testid="stChatMessage"] {{
        border-radius: 16px;
        padding: 6px 4px;
        margin-bottom: 4px;
    }}
    [data-testid="stChatMessage-user"] {{
        background: var(--user-bubble-bg) !important;
        border: 1px solid var(--user-bubble-border);
    }}
    [data-testid="stChatMessage-assistant"] {{
        background: var(--assistant-bubble-bg) !important;
        border: 1px solid var(--border);
    }}
    div[data-testid="stChatMessage"] p, div[data-testid="stChatMessage"] li {{
        overflow-wrap: anywhere;
    }}

    /* ---------- Sidebar ---------- */
    section[data-testid="stSidebar"] {{
        background: linear-gradient(180deg, var(--bg-elevated) 0%, var(--bg) 100%);
    }}
    .sidebar-brand {{
        font-size: 1.05rem;
        font-weight: 700;
        color: var(--text);
        margin-bottom: 14px;
    }}
    .sidebar-section-label {{
        font-size: 0.78rem;
        color: var(--text-muted);
        margin: 14px 0 6px 2px;
    }}
    .history-timestamp {{
        font-size: 0.72rem;
        color: var(--text-muted);
        margin: -8px 0 6px 4px;
    }}

    /* Keep tap targets >=44px on mobile without forcing sidebar width,
       so Streamlit's own mobile sidebar-collapse behavior stays intact. */
    div[data-testid="stButton"] button {{
        min-height: 44px;
    }}

    /* Focus rings use the accent identity */
    button:focus-visible, input:focus-visible, textarea:focus-visible {{
        outline: 2px solid var(--accent-mid) !important;
        outline-offset: 2px;
    }}

    /* ---------- Centered reading column (avoids full-bleed text on wide monitors) ---------- */
    .block-container {{
        max-width: 760px;
        padding-top: 2rem;
    }}

    /* ---------- Message entrance animation ---------- */
    @keyframes msg-in {{
        from {{ opacity: 0; transform: translateY(6px); }}
        to   {{ opacity: 1; transform: translateY(0); }}
    }}
    div[data-testid="stChatMessage"] {{
        animation: msg-in 220ms ease-out;
    }}

    /* ---------- Copy-to-clipboard control under assistant replies ---------- */
    .copy-btn-row {{
        display: flex;
        justify-content: flex-end;
        margin-top: -6px;
    }}
    .copy-btn {{
        font-size: 0.72rem;
        font-weight: 500;
        color: var(--text-muted);
        background: transparent;
        border: 1px solid var(--border);
        border-radius: 999px;
        padding: 2px 10px;
        cursor: pointer;
        transition: color 120ms ease, border-color 120ms ease;
    }}
    .copy-btn:hover {{
        color: var(--accent-mid);
        border-color: var(--accent-mid);
    }}
    .copy-btn.copied {{
        color: #16A34A;
        border-color: #16A34A;
    }}

    /* ---------- Sidebar footer badge ---------- */
    .sidebar-footer-badge {{
        font-size: 0.72rem;
        color: var(--text-muted);
        text-align: center;
        padding-top: 4px;
    }}

    @media (max-width: 480px) {{
        .header-banner h1 {{ font-size: 1.5rem; }}
        .landing-hero h2 {{ font-size: 1.25rem; }}
    }}
    </style>
    """,
    unsafe_allow_html=True,
)

# ============================================================
# HEADER BANNER
# ============================================================
st.markdown(
    f"""
    <div class="header-banner">
        <h1>{_spark_icon(30)} SkillSync Chatbot</h1>
        <p>Ask me anything about SkillSync -- courses, focus areas, and skillIT</p>
    </div>
    """,
    unsafe_allow_html=True,
)

# ============================================================
# SESSION STATE: things that must survive across reruns
# ============================================================
if "client" not in st.session_state:
    try:
        st.session_state.client = load_client()
    except ValueError as e:
        st.error(f"Setup required:\n\n{e}")
        st.stop()


def _blank_history(mode):
    return [{"role": "system", "content": get_system_prompt(mode)}]


def _start_new_chat(mode="qa"):
    """Begin a brand-new, unsaved conversation. Nothing is written to the
    store until the first exchange happens (autosave, see below)."""
    st.session_state.active_chat_id = str(uuid.uuid4())
    st.session_state.history = _blank_history(mode)
    st.session_state.chat_title = None
    st.session_state.mode_select = mode
    st.session_state.committed_mode = mode


def _open_chat(chat_id):
    """Load a previously saved chat into the active view."""
    messages = load_chat(chat_id)
    meta = get_chat_meta(chat_id)
    if messages is None or meta is None:
        return
    st.session_state.active_chat_id = chat_id
    st.session_state.history = messages
    st.session_state.chat_title = meta["title"]
    st.session_state.mode_select = meta["mode"]
    st.session_state.committed_mode = meta["mode"]


if "active_chat_id" not in st.session_state:
    _start_new_chat(mode="qa")

# ============================================================
# SIDEBAR: new chat, history, settings, save/clear
# ============================================================
with st.sidebar:
    st.markdown(
        f'<div class="sidebar-brand">{_spark_icon(20)} SkillSync</div>',
        unsafe_allow_html=True,
    )

    if st.button("\uFF0B  New chat", use_container_width=True, type="primary", key="new_chat_btn"):
        _start_new_chat(mode=st.session_state.mode_select)
        st.rerun()

    st.markdown('<div class="sidebar-section-label">HISTORY</div>', unsafe_allow_html=True)

    saved_chats = list_chats()
    if not saved_chats:
        st.caption("No saved chats yet -- start typing below.")
    else:
        for chat in saved_chats:
            is_active = chat["id"] == st.session_state.active_chat_id
            row_cols = st.columns([5, 1])
            with row_cols[0]:
                if st.button(
                    chat["title"] or "New chat",
                    key=f"open_{chat['id']}",
                    use_container_width=True,
                    type="primary" if is_active else "secondary",
                ):
                    _open_chat(chat["id"])
                    st.rerun()
                st.markdown(
                    f'<div class="history-timestamp">{relative_time(chat["updated_at"])}</div>',
                    unsafe_allow_html=True,
                )
            with row_cols[1]:
                if st.button("\U0001F5D1", key=f"del_{chat['id']}", help="Delete this chat"):
                    delete_chat(chat["id"])
                    if is_active:
                        _start_new_chat(mode=st.session_state.mode_select)
                    st.toast("Chat deleted", icon="\U0001F5D1")
                    st.rerun()

    st.divider()

    with st.expander("\u2699\uFE0F Settings"):
        mode = st.selectbox(
            "Assistant mode",
            options=["qa", "summarize"],
            format_func=lambda m: "\U0001F4AC Q&A Assistant" if m == "qa" else "Summarizer",
            key="mode_select",
        )
        max_tokens = st.slider(
            "Max reply length (tokens)",
            min_value=100,
            max_value=1000,
            value=500,
            step=50,
            help="Roughly, higher = longer replies but slower and more expensive.",
        )

    # Mode changed mid-conversation -> start a fresh chat, same as the
    # original app did (the two modes use different, incompatible system
    # prompts, so an existing history can't just be relabeled).
    if st.session_state.mode_select != st.session_state.committed_mode:
        _start_new_chat(mode=st.session_state.mode_select)
        st.rerun()

    transcript_lines = []
    for m in st.session_state.history:
        if m["role"] == "system":
            continue
        speaker = "You" if m["role"] == "user" else "SkillSync Bot"
        transcript_lines.append(f"{speaker}: {m['content']}")
    transcript_text = "\n\n".join(transcript_lines) or "No messages yet."

    st.download_button(
        "\u2B07\uFE0F Save chat",
        data=transcript_text,
        file_name=f"skillsync_chat_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
        use_container_width=True,
    )

    st.divider()

    # A hard "wipe everything" escape hatch, separate from New chat, per
    # the brief's rationale in section 6.1.
    if st.session_state.get("confirm_clear_all"):
        st.warning("Delete every saved chat? This can't be undone.")
        c1, c2 = st.columns(2)
        if c1.button("Yes, clear all", use_container_width=True, key="confirm_clear_yes"):
            delete_all_chats()
            _start_new_chat(mode=st.session_state.mode_select)
            st.session_state.confirm_clear_all = False
            st.toast("All chats cleared", icon="\U0001F9F9")
            st.rerun()
        if c2.button("Cancel", use_container_width=True, key="confirm_clear_cancel"):
            st.session_state.confirm_clear_all = False
            st.rerun()
    else:
        if st.button("Clear all history", use_container_width=True, key="clear_all_btn"):
            st.session_state.confirm_clear_all = True
            st.rerun()

    st.divider()
    st.markdown(
        """
        **About SkillSync**

        SkillSync is a training platform teaching AI workflows,
        automation engineering, and full-stack development through
        hands-on, real-world projects -- alongside its placement
        partner, **skillIT**.

        *All chats in this app are stored in one shared, local history
        file (no login system exists yet) -- see chat_store.py for details.*
        """
    )
    st.markdown(
        '<div class="sidebar-footer-badge">Built with Streamlit · Powered by Groq</div>',
        unsafe_allow_html=True,
    )

mode = st.session_state.mode_select

# ============================================================
# CHAT INPUT (pins to the bottom of the page regardless of call order)
# ============================================================
typed_input = st.chat_input("Type your message...")

pending_prompt = st.session_state.pop("pending_prompt", None)
user_input = pending_prompt or typed_input

USER_AVATAR = "\U0001F9D1\u200D\U0001F4BB"
BOT_AVATAR = "\U0001F9E0"

is_empty_chat = len(st.session_state.history) == 1  # only the system message so far
show_landing = user_input is None and is_empty_chat

# ============================================================
# LANDING STATE (empty chat, nothing typed yet)
# ============================================================
if show_landing:
    if not st.session_state.get("intro_shown"):
        with open(os.path.join(APP_DIR, "assets", "intro.html"), encoding="utf-8") as f:
            intro_html = f.read()
        text_color = "#F5F3FF" if IS_DARK else "#3A2E66"
        intro_html = intro_html.replace("__TEXT_COLOR__", text_color)
        components_html(intro_html, height=280, scrolling=False)
        st.session_state.intro_shown = True

    st.markdown(
        """
        <div class="landing-hero">
            <h2>What can I help you with?</h2>
            <p>I'm the SkillSync assistant -- ask me about courses, focus areas,
            the skillIT placement track, or how to get started.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    chip_cols = st.columns(len(SUGGESTED_PROMPTS))
    for col, prompt_text in zip(chip_cols, SUGGESTED_PROMPTS):
        with col:
            if st.button(prompt_text, key=f"chip_{prompt_text}", use_container_width=True):
                st.session_state.pending_prompt = prompt_text
                st.rerun()

# ============================================================
# EXISTING CONVERSATION (skip while showing the landing state)
# ============================================================
else:
    for idx, message in enumerate(st.session_state.history):
        if message["role"] == "system":
            continue
        avatar = USER_AVATAR if message["role"] == "user" else BOT_AVATAR
        with st.chat_message(message["role"], avatar=avatar):
            st.write(message["content"])
            if message["role"] == "assistant":
                st.markdown(
                    _copy_button_html(message["content"], key=f"hist_{idx}"),
                    unsafe_allow_html=True,
                )

# ============================================================
# PROCESS A NEW MESSAGE (typed, or a suggested-prompt chip)
# ============================================================
if user_input:
    with st.chat_message("user", avatar=USER_AVATAR):
        st.write(user_input)
    st.session_state.history.append({"role": "user", "content": user_input})

    if st.session_state.chat_title is None:
        st.session_state.chat_title = generate_title(user_input)

    with st.chat_message("assistant", avatar=BOT_AVATAR):
        with st.spinner("SkillSync Bot is thinking..."):
            reply = call_model(
                st.session_state.client,
                st.session_state.history,
                max_tokens=max_tokens,
            )
        st.write(reply)
        st.markdown(
            _copy_button_html(reply, key=f"latest_{len(st.session_state.history)}"),
            unsafe_allow_html=True,
        )

    st.session_state.history.append({"role": "assistant", "content": reply})

    # Autosave after every assistant reply (brief section 6.3), so closing
    # the tab never loses a conversation.
    save_chat(
        st.session_state.active_chat_id,
        st.session_state.chat_title,
        mode,
        st.session_state.history,
    )