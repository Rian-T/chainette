"""Engine configuration for the Pancreas Cancer French Clinical Note Generator pipeline."""

import os

from chainette import register_engine


def setup_engines():
    """Register all engines needed for the pipeline."""
    
    OPENAI_KEY = os.getenv("OPENAI_API_KEY", "")
    if not OPENAI_KEY:
        raise RuntimeError("OPENAI_API_KEY environment variable not set.")

    # Register OpenAI engines
    register_engine(
        "openai_mini",
        backend="openai",
        model="gpt-4.1-mini",
        api_key=OPENAI_KEY,
    )

    register_engine(
        "o4-mini",
        backend="openai",
        model="o4-mini",
        api_key=OPENAI_KEY,
    )

    register_engine(
        "gpt-4.1",
        backend="openai",
        model="gpt-4.1",
        api_key=OPENAI_KEY,
    ) 