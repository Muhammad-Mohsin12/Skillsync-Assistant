"""
cli.py
------
The main entry point for the LLM-Powered CLI Assistant.

Run it from the terminal with:
    python cli.py                  (defaults to Q&A mode)
    python cli.py --mode summarize (summarizer mode instead)

What this program does, step by step:
    1. Load your Groq API key from a .env file (never hardcoded in source).
    2. Ask which "mode" you want: a Q&A assistant or a text summarizer.
    3. Loop: read what you type, send it (plus the conversation history so
       far) to the model, print the reply.
    4. Type 'quit' (or Ctrl+C) to exit -- the full conversation is then
       automatically saved to a JSON file inside logs/.

Every risky step (missing API key, a failed network call) is handled
gracefully -- this program is designed to never crash with a raw traceback
in the user's face.
"""

import argparse
#what is argparse doing: argparse is a standard library module in Python that provides a way
#    to handle command-line arguments passed to a script. It allows you to define what arguments
#    your program requires, parse those arguments from the command line, and automatically 
# generate help and usage messages. In this code snippet, argparse is used to read
#  command-line arguments such as the mode of operation (Q&A or summarizer) and the 
# maximum number of tokens for the model's response.
import sys

from groq_client import load_client, call_model
from utils import get_system_prompt, estimate_tokens, total_tokens_in_history, save_session


def parse_args():
    """
    Read command-line arguments, e.g.:  python cli.py --mode summarize

    argparse handles turning raw sys.argv text into a clean, typed object
    (args.mode), and automatically generates a helpful --help message.
    """
    parser = argparse.ArgumentParser(
        description="A simple LLM-powered CLI assistant, powered by Groq's free API."
    )
    parser.add_argument(
        "--mode",
        choices=["qa", "summarize"],
        default="qa",
        help="Which assistant mode to run: 'qa' (default) or 'summarize'.",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=None,
        help="Override the maximum reply length (in tokens). Defaults to the "
             "MAX_TOKENS environment variable, or 500 if that isn't set either.",
    )
    return parser.parse_args()


def print_welcome(mode):
    """Print a short banner so the user knows what mode they're in and how to exit."""
    mode_label = "Q&A Assistant" if mode == "qa" else "Summarizer"
    print("=" * 50)
    print(f"  Groq CLI Assistant (made by NAME-- {mode_label} mode")
    print("=" * 50)
    print("Type your message and press Enter.")
    print("Type 'quit' (or press Ctrl+C) to exit and save this session.\n")


def run(mode="qa", max_tokens=None):
    """
    The main program loop. Broken out into its own function (separate from
    the argparse / __main__ boilerplate below) so it can also be imported
    and called directly from other scripts or tests if needed.
    """
    # ---- Step 1: load the client, with a friendly error if the key is missing ----
    try:
        client = load_client()
    except ValueError as e:
        # This is a SETUP problem (no key configured yet), not a bug -- print
        # clear instructions and exit cleanly instead of showing a traceback.
        print(f"Setup required:\n{e}")
        sys.exit(1)

    system_prompt = get_system_prompt(mode)

    # `history` is the running list of {"role": ..., "content": ...} messages.
    # We start it with just the system prompt; every user message and every
    # assistant reply gets appended to this same list as the conversation
    # continues, which is what gives the model "memory" of earlier turns
    # (the model itself has none -- WE resend the whole history every time).
    history = [{"role": "system", "content": system_prompt}]

    print_welcome(mode)

    try:
        while True:
            user_input = input("You: ").strip()

            if not user_input:
                # Ignore empty input (just pressing Enter) instead of sending
                # a blank message to the API.
                continue

            if user_input.lower() in ("quit", "exit"):
                break

            # Record the user's turn BEFORE calling the model, so the model
            # sees it as part of the conversation it's replying to.
            history.append({"role": "user", "content": user_input})

            # A lightweight awareness check: warn (don't block) if the
            # conversation is getting long, since every extra message makes
            # every future call more expensive and eventually risks hitting
            # the model's context window limit.
            token_count = total_tokens_in_history(history)
            if token_count > 3000:
                print(f"  (Note: conversation is getting long — ~{token_count} "
                      f"estimated tokens so far.)")
            

            print("Assistant: ", end="", flush=True)
            reply = call_model(client, history, max_tokens=max_tokens)
            print(reply)

            # Record the assistant's reply too, so IT is remembered on the
            # next turn as well.
            history.append({"role": "assistant", "content": reply})

    except KeyboardInterrupt:
        # Let Ctrl+C exit cleanly (with a newline so the prompt doesn't look
        # broken) instead of dumping a KeyboardInterrupt traceback.
        print("\n(Interrupted.)")

    finally:
        # This block runs whether we exited via 'quit', Ctrl+C, or even an
        # unexpected error above -- so the conversation is saved either way.
        if len(history) > 1:  # more than just the system prompt means real content happened
            filepath = save_session(history, mode)
            print(f"\nSession saved to: {filepath}")
        else:
            print("\nNo messages exchanged -- nothing to save.")


if __name__ == "__main__":
    args = parse_args()
    run(mode=args.mode, max_tokens=args.max_tokens)