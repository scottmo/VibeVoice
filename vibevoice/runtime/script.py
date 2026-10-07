"""User-facing speaker labels, normalized to the processor's 1-based script."""

import re
import secrets


def resolve_seed(value=42):
    """Zero requests a fresh seed; positive integers are repeatable requests."""
    try:
        seed = int(value)
        if seed != value or not 0 <= seed <= 2**32 - 1:
            raise ValueError
    except (TypeError, ValueError, OverflowError):
        raise ValueError("Seed must be an integer between 0 and 4294967295.") from None
    return seed or secrets.randbelow(2**32 - 1) + 1


def parse_script(script, num_speakers=None):
    """Return zero-based turns; explicit labels own subsequent continuation lines.

    Plain scripts retain line-by-line rotation. Legacy scripts containing
    Speaker 0 use zero-based long labels; bracket labels always start at 1.
    """
    lines = [line.strip() for line in script.splitlines() if line.strip()]
    zero_based = any(re.match(r"^Speaker\s+0\s*:", line, re.I) for line in lines)
    turns = []
    explicit = False
    for line in lines:
        short = re.match(r"^\[(\d+)\]\s*:?\s*(.*)$", line)
        long = re.match(r"^Speaker\s+(\d+)\s*:\s*(.*)$", line, re.I)
        match = short or long
        if match:
            label = int(match[1])
            speaker = label - (1 if short or not zero_based else 0)
            if speaker < 0 or (num_speakers is not None and speaker >= num_speakers):
                raise ValueError(f"Speaker label is outside the selected voices: {line}")
            turns.append((speaker, match[2].strip()))
            explicit = True
        elif re.match(r"^(?:\[[\d+-]|Speaker\s)", line, re.I):
            raise ValueError(f"Invalid speaker label: {line}. Use [1] Text or Speaker 1: Text.")
        elif explicit:
            speaker, text = turns[-1]
            turns[-1] = (speaker, f"{text} {line}".strip())
        else:
            turns.append((len(turns) % (num_speakers or 1), line))
    if not turns or any(not text for _, text in turns):
        raise ValueError("Provide text for every speaker turn.")
    return turns


def normalize_script(script, num_speakers):
    return "\n".join(f"Speaker {speaker + 1}: {text}" for speaker, text in parse_script(script, num_speakers))
