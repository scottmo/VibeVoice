"""Single-speaker realtime controls for the standalone Gradio app."""

import threading
from contextlib import closing

import gradio as gr
import numpy as np

from vibevoice.realtime.service import (
    SAMPLE_RATE,
    RealtimeService,
    discover_realtime_models,
)
from vibevoice.realtime.voices import discover_realtime_voices
from vibevoice.realtime.worker import RealtimeWorkerService
from vibevoice.runtime.native_select import (
    native_select_reader_js,
    render_native_select,
    select_values_js,
)
from vibevoice.runtime.script import resolve_seed

MODEL_SELECT_ID = "realtime-model-select"
VOICE_SELECT_ID = "realtime-voice-select"


class RealtimeController:
    def __init__(self, demo):
        self.demo = demo
        service_class = (
            RealtimeWorkerService if demo.load_on_demand else RealtimeService
        )
        self.service = service_class(demo.model_settings, demo.device)
        self.models = discover_realtime_models(demo.model_settings)
        self.voices = discover_realtime_voices(demo.model_settings)
        self.cancel = None
        self.owner = None

    def refresh(self, model, voice):
        self.models = discover_realtime_models(self.demo.model_settings)
        self.voices = discover_realtime_voices(self.demo.model_settings)
        return (
            self.model_select(model),
            self.voice_select(voice),
            gr.update(interactive=bool(self.models and self.voices)),
            self.asset_status(),
        )

    def model_select(self, selected=None):
        return render_native_select(
            MODEL_SELECT_ID,
            "Realtime Model",
            self.models,
            selected,
            info="Missing model files download automatically on first use."
            if self.models
            else None,
            empty_message=(
                "No realtime models found. Add a complete VibeVoice-Realtime-0.5B "
                f"checkpoint under {self.demo.model_settings.tts_dir}, then refresh."
            ),
        )

    def voice_select(self, selected=None):
        return render_native_select(
            VOICE_SELECT_ID,
            "Realtime Voice Preset",
            self.voices,
            selected,
            info="Missing official voice presets download automatically on first use."
            if self.voices
            else None,
            empty_message=(
                "No realtime voice presets found. Add cached .pt presets under "
                f"{self.demo.model_settings.models_dir / 'voices' / 'realtime'}, then refresh."
            ),
        )

    def asset_status(self):
        missing = []
        if not self.models:
            missing.append("a complete VibeVoice-Realtime-0.5B checkpoint")
        if not self.voices:
            missing.append("cached .pt voice presets")
        if missing:
            return f"Missing {' and '.join(missing)}. Add the local assets at the paths above, then refresh."
        return "Ready. Missing model, tokenizer, and official voice files download on first use."

    def unload(self):
        self.service.unload()

    def generate_realtime(
        self, model, voice, text, seed, cfg, steps, request: gr.Request = None
    ):
        stream = None
        try:
            if model not in self.models or voice not in self.voices:
                raise ValueError("Select a realtime model and cached voice preset")
            if not text or not text.strip():
                raise ValueError("Provide text for realtime speech")
            seed = resolve_seed(seed)
            cancel = threading.Event()
            self.cancel = cancel
            self.owner = request.session_hash if request else None
            # Initialize the audio transport before loading can fail.
            yield (
                None,
                gr.update(value=None, visible=False),
                f"Preparing realtime model and voice (downloading missing files)… Seed: {seed}",
            )
            if self.demo.model_loaded:
                self.demo.unload_model()
            self.service.load(self.models[model], self.voices[voice])
            if cancel.is_set():
                yield (
                    gr.update(value=None),
                    gr.update(value=None, visible=False),
                    "Realtime generation stopped.",
                )
                return
            chunks = []
            stream = self.service.stream(
                text,
                seed=seed,
                cfg_scale=cfg,
                diffusion_steps=steps,
                cancel_event=cancel,
            )
            with closing(stream):
                for chunk in stream:
                    if cancel.is_set():
                        break
                    pcm = (np.clip(chunk, -1.0, 1.0) * 32767).astype(np.int16)
                    chunks.append(pcm)
                    yield (
                        (SAMPLE_RATE, pcm),
                        gr.update(visible=False),
                        f"Streaming realtime speech… Seed: {seed}",
                    )
            if cancel.is_set():
                yield (
                    gr.update(value=None),
                    gr.update(value=None, visible=False),
                    "Realtime generation stopped.",
                )
            elif chunks:
                audio = np.concatenate(chunks)
                yield (
                    None,
                    gr.update(value=(SAMPLE_RATE, audio), visible=True),
                    f"Realtime speech complete: {audio.size / SAMPLE_RATE:.1f}s. Seed: {seed}",
                )
            else:
                raise RuntimeError("Realtime model produced no audio")
        except Exception as exc:  # noqa: BLE001 - surface failures in the UI
            yield (
                None,
                gr.update(value=None, visible=False),
                f"Realtime generation failed: {exc}",
            )
        finally:
            if stream is not None:
                stream.close()
            self.cancel = None
            self.owner = None

    def stop_realtime(self, request: gr.Request = None):
        if self.cancel is not None and (
            request is None or self.owner == request.session_hash
        ):
            self.cancel.set()
            self.service.stop()
            return (
                gr.update(value=None),
                gr.update(value=None, visible=False),
                "Stopping realtime generation…",
            )
        return gr.update(), gr.update(), gr.update()

    def disconnect(self, request: gr.Request):
        if (
            request is not None
            and self.owner == request.session_hash
            and self.cancel is not None
        ):
            self.cancel.set()
            self.service.stop()


def build_realtime_controls(demo, interface):
    controller = RealtimeController(demo)
    with gr.Accordion("Realtime TTS — Single Speaker", open=False):
        gr.Markdown(
            "Generate speech with the realtime 0.5B model and a cached voice preset. Missing assets download on first use. Audio plays while generation runs."
        )
        with gr.Row():
            model = gr.HTML(
                controller.model_select(), elem_id="realtime-model-select-field"
            )
            voice = gr.HTML(
                controller.voice_select(), elem_id="realtime-voice-select-field"
            )
        refresh = gr.Button("Refresh Realtime Assets")
        text = gr.Textbox(
            label="Realtime Text", lines=5, placeholder="Enter single-speaker text"
        )
        with gr.Row():
            seed = gr.Number(
                value=42,
                precision=0,
                minimum=0,
                label="Realtime Seed",
                info="0 chooses a random seed",
            )
            cfg = gr.Slider(1.5, 3.0, value=1.5, step=0.1, label="Realtime CFG")
            steps = gr.Slider(1, 50, value=5, step=1, label="Realtime Diffusion Steps")
        with gr.Row():
            generate = gr.Button(
                "Generate Realtime Speech",
                variant="primary",
                interactive=bool(controller.models and controller.voices),
            )
            stop = gr.Button("Stop Realtime Speech")
        live = gr.Audio(
            label="Realtime Live Audio", streaming=True, autoplay=True, type="numpy"
        )
        complete = gr.Audio(
            label="Realtime Complete Audio",
            visible=False,
            type="numpy",
            show_download_button=True,
        )
        status = gr.Textbox(
            label="Realtime Status", value=controller.asset_status(), interactive=False
        )
        refresh.click(
            controller.refresh,
            inputs=[model, voice],
            outputs=[model, voice, generate, status],
            js=select_values_js([MODEL_SELECT_ID, VOICE_SELECT_ID], [0, 1]),
            concurrency_id="model_operations",
            concurrency_limit=1,
        )
        generation = generate.click(
            controller.generate_realtime,
            inputs=[model, voice, text, seed, cfg, steps],
            outputs=[live, complete, status],
            js=(
                f"(...values) => {{ {native_select_reader_js([MODEL_SELECT_ID, VOICE_SELECT_ID], [0, 1])} "
                "return [...selected, ...values.slice(2)]; }"
            ),
            concurrency_id="model_operations",
            concurrency_limit=1,
        )
        stop.click(
            controller.stop_realtime,
            inputs=[],
            outputs=[live, complete, status],
            queue=False,
            cancels=[generation],
        )
    if hasattr(interface, "unload"):
        interface.unload(controller.disconnect)
    return controller
