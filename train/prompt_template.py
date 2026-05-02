"""Prompt template for terminal-output cleanup task. See spec §6.4."""

INSTRUCTION = (
    "Clean the following terminal output. Preserve all information losslessly. "
    "Strip ANSI codes, collapse progress-bar overwrites to their final state, "
    "deduplicate identical repeated lines using [Nx] prefix, normalize whitespace."
)


def format_chat(dirty: str, clean: str) -> dict:
    """Return MLX-LM-compatible chat-format record."""
    return {
        "messages": [
            {"role": "user", "content": f"{INSTRUCTION}\n\n---\n{dirty}\n---"},
            {"role": "assistant", "content": clean},
        ],
    }
