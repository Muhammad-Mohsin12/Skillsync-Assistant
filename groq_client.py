"""
groq_client.py
--------------
Everything related to talking to the Groq API lives in this one file:
  1. Loading the API key safely from a .env file (never hardcoded)
  2. Initializing the Groq client
  3. A single "call_model" function that every other part of the project uses
     to actually send messages to the model and get a reply back -- with
     built-in error handling AND automatic retries for flaky network issues.

Keeping all of this in one file means the rest of the project (cli.py) never
has to think about API details -- it just calls call_model() and gets a
plain string back, every time, no matter what went wrong under the hood.
"""

import os
import time
from dotenv import load_dotenv
from groq import Groq


def load_client():
    load_dotenv()

    api_key = os.getenv("GROQ_API_KEY")

    if not api_key:
        raise ValueError(
            "GROQ_API_KEY not found.\n"
            "  1. Copy .env.example to .env\n"
            "  2. Get a free key at https://console.groq.com\n"
            "  3. Paste it into .env as GROQ_API_KEY=gsk_your_key_here"
        )

    return Groq(api_key=api_key)


def call_model(client, messages, model=None, max_tokens=None, max_retries=3):
    """
    Send a list of chat messages to the Groq API and return the reply text.

    This function is intentionally the ONLY place in the whole project that
    talks to the network. Centralizing it here means:
      - error handling only needs to be written once
      - retry logic only needs to be written once
      - if we ever swap providers (e.g. to OpenAI), only this file changes

    Args:
        client: a Groq client object, as returned by load_client().
        messages (list[dict]): the conversation so far, each item shaped like
            {"role": "system" | "user" | "assistant", "content": "..."}.
        model (str): which model to use. Falls back to the DEFAULT_MODEL
            environment variable, or a sensible hardcoded default if that's
            not set either.
        max_tokens (int): the maximum length of the model's reply. Falls back
            to the MAX_TOKENS environment variable, or 500 by default.
        max_retries (int): how many times to retry on a TRANSIENT failure
            (timeouts, connection drops, rate limits) before giving up.

    Returns:
        str: the model's reply text, OR a clearly-labeled error message
             string starting with "[Error" if every retry failed. This
             function deliberately never raises -- callers can always safely
             print() or log whatever it returns without wrapping every call
             in their own try/except.
    """
    # Resolve model/max_tokens from arguments -> environment variables -> hardcoded defaults.
    # This three-level fallback is a common, flexible configuration pattern.
    model = model or os.getenv("DEFAULT_MODEL", "qwen/qwen3.8-27b")
    max_tokens = max_tokens or int(os.getenv("MAX_TOKENS", "500"))

    last_error = None

    for attempt in range(1, max_retries + 1):
        try:
            response = client.chat.completions.create(
                model=model,
                max_tokens=max_tokens,
                messages=messages,
            )
            # This is where we dig into the response object to get the
            # actual text. Groq mirrors OpenAI's response shape:
            #   response.choices[0].message.content
            return response.choices[0].message.content

        except Exception as e:
            # We deliberately catch a broad Exception here (rather than
            # listing every specific Groq/network exception type) because
            # different versions of the SDK and different failure modes
            # (timeouts, connection errors, rate limits, bad JSON, etc.)
            # can all surface as different exception classes. What matters
            # for a CLI tool is: something went wrong, log it, maybe retry.
            last_error = e
            print(f"  [attempt {attempt}/{max_retries}] API call failed: {e}")

            if attempt < max_retries:
                # Exponential backoff: wait longer after each failed attempt
                # (1s, 2s, 4s, ...) so we don't hammer an already-struggling
                # or rate-limited service.
                wait_seconds = 2 ** (attempt - 1)
                time.sleep(wait_seconds)

    # If we reach this point, every single attempt failed.
    return f"[Error: could not get a response after {max_retries} attempts -- {last_error}]"