"""
app.py
------
The web-based frontend for the "SkillSync Chatbot", built with Streamlit.

This file does NOT reinvent any of the logic from cli.py, groq_client.py, or
utils.py -- it reuses all of it. Everything new in THIS file is purely visual:
colors, layout, avatars, and a couple of small extra buttons. The "brain"
(how the model is called, how errors are handled) is exactly the same as
before.

HOW TO RUN THIS FILE:
    python -m streamlit run app.py

FIRST TIME ONLY -- install streamlit before running:
    python -m pip install streamlit
"""

import streamlit as st
from datetime import datetime

from groq_client import load_client, call_model
from utils import get_system_prompt


# ============================================================
# PAGE SETUP
# ============================================================
# This controls the browser tab's title + the little icon shown on the tab.
st.set_page_config(
    page_title="SkillSync Assistant",
    page_icon="🧠",
    layout="centered",
)

# ============================================================
# CUSTOM STYLING (CSS)
# ============================================================
# Streamlit lets you inject plain CSS to restyle things it doesn't give you
# a built-in option for. unsafe_allow_html=True is required for this to work
# -- it's "unsafe" only in the sense that Streamlit trusts you not to inject
# something malicious here; since WE wrote this text ourselves, it's fine.
st.markdown(
    """
    <style>
    /* The big gradient header banner at the top of the page */
    .header-banner {
        background: linear-gradient(90deg, #4F46E5 0%, #7C3AED 50%, #DB2777 100%);
        padding: 28px 24px;
        border-radius: 16px;
        margin-bottom: 24px;
        text-align: center;
    }
    .header-banner h1 {
        color: white;
        font-size: 2rem;
        margin: 0;
    }
    .header-banner p {
        color: #EDE9FE;
        margin-top: 6px;
        font-size: 0.95rem;
    }

    /* Give chat bubbles a bit more breathing room and rounder corners */
    div[data-testid="stChatMessage"] {
        border-radius: 14px;
        padding: 4px 2px;
    }

    /* Sidebar background tweak */
    section[data-testid="stSidebar"] {
        background-color: #F5F3FF;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ============================================================
# HEADER BANNER
# ============================================================
st.markdown(
    """
    <div class="header-banner">
        <h1>🧠 SkillSync Chatbot</h1>
        <p>Ask me anything about SkillSync -- courses, focus areas, and skillIT</p>
    </div>
    """,
    unsafe_allow_html=True,
)

# ============================================================
# SIDEBAR: settings + info panel
# ============================================================
with st.sidebar:
    st.markdown("## ⚙️ Settings")

    mode = st.selectbox(
        "Assistant mode",
        options=["qa", "summarize"],
        format_func=lambda m: "💬 Q&A Assistant" if m == "qa" else "📝 Summarizer",
    )

    max_tokens = st.slider(
        "Max reply length (tokens)",
        min_value=100,
        max_value=1000,
        value=500,
        step=50,
        help="Roughly, higher = longer replies but slower and more expensive.",
    )

    st.divider()

    col1, col2 = st.columns(2)
    with col1:
        clear_clicked = st.button("🗑️ Clear chat", use_container_width=True)
    with col2:
        # st.download_button needs actual text content to offer as a file,
        # so we build a simple readable transcript from the history below.
        transcript_lines = []
        for m in st.session_state.get("history", []):
            if m["role"] == "system":
                continue
            speaker = "You" if m["role"] == "user" else "SkillSync Bot"
            transcript_lines.append(f"{speaker}: {m['content']}")
        transcript_text = "\n\n".join(transcript_lines) or "No messages yet."

        st.download_button(
            "⬇️ Save chat",
            data=transcript_text,
            file_name=f"skillsync_chat_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
            use_container_width=True,
        )

    st.divider()
    st.markdown(
        """
        **About SkillSync**

        SkillSync is a training platform teaching AI workflows,
        automation engineering, and full-stack development through
        hands-on, real-world projects -- alongside its placement
        partner, **skillIT**.
        """
    )

if clear_clicked:
    st.session_state.clear()
    st.rerun()

# ============================================================
# SESSION STATE: things that must survive across reruns
# ============================================================
# Streamlit re-runs this whole file top-to-bottom on every interaction.
# st.session_state is the one place that does NOT get wiped each time --
# it's where we keep the Groq client and the conversation history alive,
# the same role `history` played inside the loop in cli.py.

if "client" not in st.session_state:
    try:
        st.session_state.client = load_client()
    except ValueError as e:
        st.error(f"Setup required:\n\n{e}")
        st.stop()

if "history" not in st.session_state or st.session_state.get("mode") != mode:
    # Fresh history whenever the app first loads, or whenever the sidebar
    # mode is switched (qa and summarize use different system prompts).
    st.session_state.history = [
        {"role": "system", "content": get_system_prompt(mode)}
    ]
    st.session_state.mode = mode

# ============================================================
# CHAT DISPLAY
# ============================================================
# Custom avatars just for a nicer look -- purely cosmetic.
USER_AVATAR = "🧑‍💻"
BOT_AVATAR = "🧠"

if len(st.session_state.history) == 1:
    # Only the system message exists -- nothing typed yet.
    st.info("👋 Say hello, or ask something like **'What is SkillSync?'**")

for message in st.session_state.history:
    if message["role"] == "system":
        continue
    avatar = USER_AVATAR if message["role"] == "user" else BOT_AVATAR
    with st.chat_message(message["role"], avatar=avatar):
        st.write(message["content"])

# ============================================================
# CHAT INPUT
# ============================================================
user_input = st.chat_input("Type your message...")

if user_input:
    with st.chat_message("user", avatar=USER_AVATAR):
        st.write(user_input)
    st.session_state.history.append({"role": "user", "content": user_input})

    with st.chat_message("assistant", avatar=BOT_AVATAR):
        with st.spinner("SkillSync Bot is thinking..."):
            reply = call_model(
                st.session_state.client,
                st.session_state.history,
                max_tokens=max_tokens,
            )
        st.write(reply)

    st.session_state.history.append({"role": "assistant", "content": reply})
