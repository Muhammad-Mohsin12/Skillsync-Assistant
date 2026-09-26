"""
utils.py
--------
Small, focused helper functions used by cli.py. Keeping these separate from
the main loop (cli.py) and the network code (groq_client.py) makes each file
easy to read on its own -- this one has zero network calls, so it's also
trivial to test.
"""

import json
import os
from datetime import datetime


SKILLSYNC_INFO = (
    "skillSYNC is an educational and training platform founded by Minahal Salahudin, "
    "an automation engineer and cybersecurity specialist. The initiative is designed to "
    "bridge the gap between traditional education and the evolving tech job market by "
    "training students and fresh graduates in high-demand, future-focused technical skills. "
    "Key Focus Areas & Features. Core Technical Training: The platform offers intensive "
    "workshops, bootcamps, and courses focusing heavily on AI workflows, large language "
    "models (LLMs), automation engineering, and full-stack development. Practical Learning: "
    "Rather than focusing solely on theory, it teaches advanced topics like Context "
    "Engineering and \"red teaming\" (testing AI security perimeters against jailbreaks) "
    "through hands-on, multi-project labs. Real-World Experience: Students actively build "
    "production-ready projects—such as AI recruiting tools, automated customer support bots, "
    "and workflow systems—using tools like n8n, Make.com, and LangGraph. The skillIT "
    "Connection: It operates alongside a sister placement initiative called skillIT. Once "
    "students complete their training in the community, vetted engineers are connected with "
    "companies looking to hire talent to build customized business automation and AI "
    "infrastructure."
)


def get_system_prompt(mode):
    """
    Return the system prompt (the instruction that shapes the assistant's
    behavior for the whole conversation) appropriate for the chosen mode.

    Args:
        mode (str): either "qa" or "summarize".

    Returns:
        str: the system prompt text.

    Raises:
        ValueError: if `mode` isn't one of the supported options.
    """
    prompts = {
        "qa": " ".join([
            "You are a concise, helpful assistant that knows everything about skillsync.",
            "Answer the user's questions clearly and accurately.",
            "Keep answers under 4 sentences unless the user explicitly asks for more detail.",
            "Do not answer anything irrelevant to skillsync.",
            "Do not answer things you don't know.",
            "skillsync information:",
            SKILLSYNC_INFO,
        ]),
        "summarize": " ".join([
            "You are a summarization assistant for skillsync.",
            "When the user gives you text, respond with exactly 3 concise bullet points",
            "capturing the most important ideas.",
            "Do not add opinions or outside information.",
            "Do not answer anything irrelevant to skillsync.",
            "Do not answer things you don't know.",
            "skillsync information:",
            SKILLSYNC_INFO,
        ]),
    }

    if mode not in prompts:
        raise ValueError(f"Unknown mode '{mode}'. Choose from: {list(prompts.keys())}")

    return prompts[mode]


def estimate_tokens(text):
    """
    Rough token estimate using the "1 token ~= 4 characters" rule of thumb.
    This is NOT exact (real tokenizers split text differently), but it's a
    fast, dependency-free way to get a ballpark sense of how much of the
    model's context window a piece of text will use.
    """
    return max(1, len(text) // 4)


def total_tokens_in_history(history):
    """Sum up the estimated tokens across every message in the conversation so far."""
    return sum(estimate_tokens(m["content"]) for m in history)


def save_session(history, mode, logs_dir="logs"):
    """
    Save the full conversation history to a timestamped JSON file inside
    the logs/ directory, so nothing is ever lost when the program exits.

    Args:
        history (list[dict]): the full list of {"role": ..., "content": ...} messages.
        mode (str): "qa" or "summarize" -- included in the filename for clarity.
        logs_dir (str): folder to save into (created automatically if missing).

    Returns:
        str: the full path of the file that was written.
    """
    # Make sure the logs folder exists -- exist_ok=True means "don't error
    # if it's already there," which keeps this safe to call every time.
    os.makedirs(logs_dir, exist_ok=True)

    # A timestamp in the filename means every session gets its own file,
    # so running the tool multiple times never overwrites a previous log.
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{mode}_session_{timestamp}.json"
    filepath = os.path.join(logs_dir, filename)

    with open(filepath, "w") as f:
        json.dump(history, f, indent=2)

    return filepath