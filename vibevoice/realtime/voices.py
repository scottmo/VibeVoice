"""Official cached voice presets and local overrides."""

import shutil
import tempfile
from pathlib import Path
from urllib.request import urlopen

VOICE_REVISION = "1541f590c7099820f10ea012f48d2399282df69f"
VOICE_BASE_URL = (
    f"https://raw.githubusercontent.com/microsoft/VibeVoice/{VOICE_REVISION}"
    "/demo/voices/streaming_model"
)
OFFICIAL_VOICES = (
    "en-Carter_man",
    "en-Davis_man",
    "en-Emma_woman",
    "en-Frank_man",
    "en-Grace_woman",
    "en-Mike_man",
    "de-Spk0_man",
    "de-Spk1_woman",
    "fr-Spk0_man",
    "fr-Spk1_woman",
    "in-Samuel_man",
    "it-Spk0_woman",
    "it-Spk1_man",
    "jp-Spk0_man",
    "jp-Spk1_woman",
    "kr-Spk0_woman",
    "kr-Spk1_man",
    "nl-Spk0_man",
    "nl-Spk1_woman",
    "pl-Spk0_man",
    "pl-Spk1_woman",
    "pt-Spk0_woman",
    "pt-Spk1_man",
    "sp-Spk0_woman",
    "sp-Spk1_man",
)


def discover_realtime_voices(settings):
    root = settings.models_dir / "voices" / "realtime"
    voices = {name: name for name in OFFICIAL_VOICES}
    voices.update(
        {
            path.relative_to(root).with_suffix("").as_posix(): str(path.resolve())
            for path in sorted(root.rglob("*.pt"))
            if path.is_file()
        }
    )
    return voices


def resolve_voice_preset(selection, settings):
    if str(selection) not in OFFICIAL_VOICES:
        path = Path(selection).expanduser().resolve()
        if not path.is_file():
            raise FileNotFoundError(f"Missing realtime voice preset: {path}")
        return path
    destination = settings.models_dir / "voices" / "realtime" / f"{selection}.pt"
    if destination.is_file():
        return destination
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(
            dir=destination.parent, delete=False
        ) as output:
            temporary = Path(output.name)
            with urlopen(f"{VOICE_BASE_URL}/{selection}.pt", timeout=60) as response:
                shutil.copyfileobj(response, output)
        if temporary.stat().st_size == 0:
            raise ValueError(f"Downloaded realtime voice preset is empty: {selection}")
        temporary.replace(destination)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
    return destination
