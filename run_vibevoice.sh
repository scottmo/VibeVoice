#!/bin/sh
set -u

SCRIPT_DIR=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd) || {
    printf '%s\n' "Error: could not locate the VibeVoice directory." >&2
    exit 1
}

cd "$SCRIPT_DIR" || exit 1

printf 'Starting VibeVoice from %s...\n' "$SCRIPT_DIR"

if [ ! -x "venv/bin/python" ]; then
    printf '%s\n' "Error: project virtual environment not found." >&2
    printf '%s\n' "Create it with Python 3.11 and install the project dependencies first:" >&2
    printf '%s\n' "  python3 -m venv venv" >&2
    printf '%s\n' "  venv/bin/python -m pip install -r requirements.txt" >&2
    exit 1
fi

if [ ! -f ".env" ]; then
    printf '%s\n' "Warning: .env file not found. Copy .env-sample to .env to configure model loading."
fi

printf '%s\n' "Launching VibeVoice..."
printf '%s\n' "The local interface is available at http://localhost:7590" ""

exec "$SCRIPT_DIR/venv/bin/python" "$SCRIPT_DIR/main.py" "$@"
