"""
VibeVoice Gradio Demo - High-Quality Dialogue Generation Interface with Streaming Support
"""

import argparse
import json
import os
import time
from pathlib import Path
from typing import Iterator, Tuple, Optional

# Load the project .env before importing Transformers/Hugging Face modules,
# which read HF_* environment switches during import.
from vibevoice.runtime.model_loading import (
    DEFAULT_MODEL_REPOSITORIES,
    ModelLoadingSettings,
    default_model_name,
    discover_local_models,
    load_model_and_processor,
    normalize_model_selection,
    add_model_cli_arguments,
    launch_compatibly,
    settings_from_args,
)

import threading
import multiprocessing
import queue
import signal
import numpy as np
import gradio as gr
import librosa
import soundfile as sf
import torch
import traceback

# Check Gradio version for compatibility
try:
    GRADIO_VERSION = tuple(map(int, gr.__version__.split('.')[:2]))  # (major, minor)
    GRADIO_HAS_SHOW_DOWNLOAD = GRADIO_VERSION < (6, 0)  # show_download_button removed in 6.0
except:
    GRADIO_HAS_SHOW_DOWNLOAD = True  # Default to True for safety

# Device detection and attention mechanism fallback
def detect_device():
    """Detect the best available device (CUDA, MPS, or CPU)"""
    if torch.cuda.is_available():
        return "cuda", torch.cuda.get_device_name(0)
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        return "mps", "Apple Silicon (MPS)"
    else:
        return "cpu", "CPU"

def get_attention_implementation(device_type: str):
    """Get the best available attention implementation for the device"""
    if device_type == "cuda":
        try:
            # Try to import flash_attn to check if it's available
            import flash_attn
            return "flash_attention_2"
        except ImportError:
            print("⚠️ FlashAttention2 not available, falling back to SDPA")
            return "sdpa"
    elif device_type == "mps":
        # Apple Silicon doesn't support flash_attention_2, use SDPA
        return "sdpa"
    else:
        # CPU fallback to SDPA
        return "sdpa"

from vibevoice.modular.streamer import AudioStreamer
from vibevoice.utils.vocal_isolation import VocalIsolator, clear_vocal_isolator_cache
from transformers.utils import logging
from transformers import set_seed
from vibevoice.runtime.script import normalize_script, resolve_seed
from vibevoice.asr.service import asr_python, discover_asr_models, run_transcription
from vibevoice.realtime.ui import build_realtime_controls
from vibevoice.runtime.native_select import (
    native_select_reader_js,
    render_native_select,
    select_values_js,
)

logging.set_verbosity_info()
logger = logging.get_logger(__name__)


# Audio Trimming Functions
def create_waveform_plot(audio_data: np.ndarray, start_time: float = 0, end_time: float = None, sample_rate: int = 24000) -> str:
    """Create a waveform visualization with trim markers"""
    if audio_data is None or len(audio_data) == 0:
        return None
        
    # Handle Gradio Audio component format (sample_rate, audio_data) tuple
    if isinstance(audio_data, tuple) and len(audio_data) == 2:
        sample_rate, audio_data = audio_data
        
    # Ensure audio_data is a numpy array
    if not isinstance(audio_data, np.ndarray):
        audio_data = np.array(audio_data)
        
    duration = len(audio_data) / sample_rate
    if end_time is None:
        end_time = duration
        
    # Create the plot
    fig, ax = plt.subplots(figsize=(12, 4))
    
    # Plot the full waveform
    time_axis = np.linspace(0, duration, len(audio_data))
    ax.plot(time_axis, audio_data, color='#22c55e', linewidth=0.5, alpha=0.7)
    ax.fill_between(time_axis, audio_data, alpha=0.3, color='#22c55e')
    
    # Add trim markers
    ax.axvline(x=start_time, color='#ef4444', linewidth=2, linestyle='--', alpha=0.8, label=f'Start: {start_time:.2f}s')
    ax.axvline(x=end_time, color='#ef4444', linewidth=2, linestyle='--', alpha=0.8, label=f'End: {end_time:.2f}s')
    
    # Highlight the selected region
    ax.axvspan(start_time, end_time, alpha=0.2, color='#3b82f6', label='Selected Region')
    
    # Styling
    ax.set_xlabel('Time (seconds)')
    ax.set_ylabel('Amplitude')
    ax.set_title('Audio Waveform with Trim Markers')
    ax.grid(True, alpha=0.3)
    ax.legend()
    ax.set_xlim(0, duration)
    
    # Convert to base64 string for Gradio
    buffer = io.BytesIO()
    plt.savefig(buffer, format='png', dpi=100, bbox_inches='tight')
    buffer.seek(0)
    image_base64 = base64.b64encode(buffer.getvalue()).decode()
    plt.close(fig)
    
    return f"data:image/png;base64,{image_base64}"


def apply_gain_to_audio(audio_data: np.ndarray, gain_db: float, sample_rate: int = 24000) -> np.ndarray:
    """Apply gain adjustment to audio data"""
    if audio_data is None or len(audio_data) == 0:
        return audio_data
        
    # Handle Gradio Audio component format (sample_rate, audio_data) tuple
    if isinstance(audio_data, tuple) and len(audio_data) == 2:
        sample_rate, audio_data = audio_data
        
    # Ensure audio_data is a numpy array
    if not isinstance(audio_data, np.ndarray):
        audio_data = np.array(audio_data)
    
    # Convert dB to linear gain
    linear_gain = 10 ** (gain_db / 20.0)
    
    # Apply gain
    gained_audio = audio_data * linear_gain
    
    # Prevent clipping by normalizing if needed
    max_val = np.max(np.abs(gained_audio))
    if max_val > 1.0:
        gained_audio = gained_audio / max_val
    
    return gained_audio


def trim_and_apply_gain(audio_data: np.ndarray, start_time: float, end_time: float, gain_db: float, sample_rate: int = 24000) -> Tuple[np.ndarray, str]:
    """Trim audio data and apply gain adjustment"""
    if audio_data is None or len(audio_data) == 0:
        return None, "No audio data to trim"
        
    # Handle Gradio Audio component format (sample_rate, audio_data) tuple
    if isinstance(audio_data, tuple) and len(audio_data) == 2:
        sample_rate, audio_data = audio_data
        
    # Ensure audio_data is a numpy array
    if not isinstance(audio_data, np.ndarray):
        audio_data = np.array(audio_data)
        
    duration = len(audio_data) / sample_rate
    
    # Validate trim times
    start_time = max(0, start_time)
    end_time = min(duration, end_time)
    
    if start_time >= end_time:
        return None, "Invalid trim range: start time must be less than end time"
        
    # Convert to sample indices
    start_sample = int(start_time * sample_rate)
    end_sample = int(end_time * sample_rate)
    
    # Trim the audio
    trimmed_audio = audio_data[start_sample:end_sample]
    
    # Apply gain if not zero
    if gain_db != 0:
        trimmed_audio = apply_gain_to_audio(trimmed_audio, gain_db, sample_rate)
    
    # Create info string
    original_duration = duration
    trimmed_duration = len(trimmed_audio) / sample_rate
    gain_text = f" (gain: {gain_db:+.1f}dB)" if gain_db != 0 else ""
    info = f"Trimmed: {original_duration:.2f}s → {trimmed_duration:.2f}s (removed {original_duration - trimmed_duration:.2f}s){gain_text}"
    
    return trimmed_audio, info


def create_trim_preview_audio(audio_data: np.ndarray, start_time: float, end_time: float, gain_db: float = 0, sample_rate: int = 24000) -> Tuple[np.ndarray, str]:
    """Create preview audio for the selected trim region"""
    if audio_data is None:
        return None, "No audio data"
        
    # Handle Gradio Audio component format (sample_rate, audio_data) tuple
    if isinstance(audio_data, tuple) and len(audio_data) == 2:
        sample_rate, audio_data = audio_data
        
    # Ensure audio_data is a numpy array
    if not isinstance(audio_data, np.ndarray):
        audio_data = np.array(audio_data)
        
    duration = len(audio_data) / sample_rate
    
    # Validate trim times
    start_time = max(0, start_time)
    end_time = min(duration, end_time)
    
    if start_time >= end_time:
        return None, "Invalid trim range"
        
    # Convert to sample indices
    start_sample = int(start_time * sample_rate)
    end_sample = int(end_time * sample_rate)
    
    # Extract preview audio
    preview_audio = audio_data[start_sample:end_sample]
    
    # Apply gain if not zero
    if gain_db != 0:
        preview_audio = apply_gain_to_audio(preview_audio, gain_db, sample_rate)
    
    preview_duration = len(preview_audio) / sample_rate
    gain_text = f" (gain: {gain_db:+.1f}dB)" if gain_db != 0 else ""
    info = f"Preview: {preview_duration:.2f}s{gain_text}"
    
    return preview_audio, info


def update_audio_trimmer(audio_data: np.ndarray) -> Tuple[str, float, float, float, str]:
    """Update the audio trimmer interface when new audio is loaded"""
    if audio_data is None:
        return None, 0, 0, 0, "No audio loaded"
    
    # Handle Gradio Audio component format (sample_rate, audio_data) tuple
    if isinstance(audio_data, tuple) and len(audio_data) == 2:
        sample_rate, audio_data = audio_data
    else:
        sample_rate = 24000
        
    if len(audio_data) == 0:
        return None, 0, 0, 0, "No audio loaded"
    
    duration = len(audio_data) / sample_rate
    waveform = create_waveform_plot(audio_data, 0, duration, sample_rate)
    return waveform, duration, 0, duration, f"Audio loaded: {duration:.2f}s duration"


def update_waveform_with_trim(audio_data: np.ndarray, start_time: float, end_time: float) -> str:
    """Update waveform visualization when trim markers change"""
    if audio_data is None:
        return None
    
    # Handle Gradio Audio component format (sample_rate, audio_data) tuple
    if isinstance(audio_data, tuple) and len(audio_data) == 2:
        sample_rate, audio_data = audio_data
    else:
        sample_rate = 24000
    
    waveform = create_waveform_plot(audio_data, start_time, end_time, sample_rate)
    return waveform


def apply_audio_trim(audio_data: np.ndarray, start_time: float, end_time: float, gain_db: float = 0) -> Tuple[np.ndarray, str]:
    """Apply trim to audio and return trimmed audio with info"""
    if audio_data is None:
        return None, "No audio data to trim"
    
    # Handle Gradio Audio component format (sample_rate, audio_data) tuple
    if isinstance(audio_data, tuple) and len(audio_data) == 2:
        sample_rate, audio_data = audio_data
    else:
        sample_rate = 24000
    
    trimmed_audio, info = trim_and_apply_gain(audio_data, start_time, end_time, gain_db, sample_rate)
    return trimmed_audio, info


def reset_audio_trim(audio_data: np.ndarray) -> Tuple[str, float, float, str]:
    """Reset trim markers to full audio"""
    if audio_data is None:
        return None, 0, 0, "No audio loaded"
    
    # Handle Gradio Audio component format (sample_rate, audio_data) tuple
    if isinstance(audio_data, tuple) and len(audio_data) == 2:
        sample_rate, audio_data = audio_data
    else:
        sample_rate = 24000
        
    if len(audio_data) == 0:
        return None, 0, 0, "No audio loaded"
    
    duration = len(audio_data) / sample_rate
    waveform = create_waveform_plot(audio_data, 0, duration, sample_rate)
    return waveform, 0, duration, f"Reset to full audio: {duration:.2f}s"


# Global variables to store original audio for gain processing
_original_audio_cache = None
_current_gain_db = 0.0
_last_cached_audio_hash = None

def cache_original_audio(audio_data: np.ndarray) -> None:
    """Cache the current audio data for gain processing (updates when audio is trimmed)"""
    global _original_audio_cache, _current_gain_db, _last_cached_audio_hash
    if audio_data is not None:
        if isinstance(audio_data, tuple) and len(audio_data) == 2:
            sample_rate, audio_data = audio_data
            
            # Create a hash of the audio to detect if it's different from what we cached
            audio_hash = hash(audio_data.tobytes())
            
            # Only update cache if this is different audio (not just gain-adjusted version)
            if audio_hash != _last_cached_audio_hash:
                # Convert to float32 but preserve original levels (don't normalize)
                if audio_data.dtype != np.float32:
                    # Convert int16 to float32 and normalize to [-1, 1] range
                    if audio_data.dtype == np.int16:
                        audio_data = audio_data.astype(np.float32) / 32767.0
                    else:
                        audio_data = audio_data.astype(np.float32)
                _original_audio_cache = (sample_rate, audio_data)
                _current_gain_db = 0.0  # Reset gain when audio changes (including trimming)
                _last_cached_audio_hash = audio_hash
        else:
            _original_audio_cache = None

def apply_gain_to_complete_audio(audio_data: np.ndarray, gain_db: float) -> Tuple[np.ndarray, str]:
    """Apply gain to the original cached audio (prevents compounding gain issues)"""
    global _original_audio_cache, _current_gain_db
    
    if _original_audio_cache is None:
        return None, "No original audio cached"
    
    sample_rate, original_audio = _original_audio_cache
    
    # Apply gain to the original audio (not the current audio)
    gain_linear = 10 ** (gain_db / 20.0)
    adjusted_audio = original_audio * gain_linear
    
    # Debug info
    original_max = np.max(np.abs(original_audio))
    adjusted_max = np.max(np.abs(adjusted_audio))
    
    # Convert back to int16 for Gradio compatibility
    # The original audio should already be in [-1, 1] range from caching
    # Apply soft clipping to prevent harsh artifacts
    adjusted_audio = np.tanh(adjusted_audio)  # Soft clipping to prevent harsh artifacts
    
    # Convert to int16 with proper scaling
    adjusted_audio = (adjusted_audio * 32767).astype(np.int16)
    
    # Update current gain
    _current_gain_db = gain_db
    
    gain_text = f" (Gain: {gain_db:+.1f}dB)" if gain_db != 0 else ""
    info = f"Audio with gain applied{gain_text} | Original max: {original_max:.3f} → Adjusted max: {adjusted_max:.3f}"
    
    return (sample_rate, adjusted_audio), info


def reset_gain_control() -> float:
    """Reset gain control to 0"""
    return 0.0


# ============================================================================
# Multiprocessing Model Worker (for true VRAM cleanup in LOD mode)
# ============================================================================

def model_worker_process(request_queue, response_queue, model_path, device, inference_steps,
                         model_settings, attn_implementation):
    """
    Worker process that loads and runs the model.
    When this process is killed, the OS forcibly reclaims ALL GPU memory.
    """
    import sys
    import os
    
    # Debug: Print Python executable and path
    print(f"[Worker] Python executable: {sys.executable}")
    print(f"[Worker] Python path (first 3): {sys.path[:3]}")
    
    # Ensure the worker can find the vibevoice package
    # The package might be installed in the parent directory
    script_dir = os.path.dirname(os.path.abspath(__file__))
    parent_dir = os.path.dirname(script_dir)
    
    # Add both script dir and parent dir to path
    for path in [script_dir, parent_dir]:
        if path not in sys.path:
            sys.path.insert(0, path)
    
    print(f"[Worker] Added to sys.path: {script_dir}")
    print(f"[Worker] Working directory: {os.getcwd()}")
    
    try:
        # Import necessary packages in worker
        import torch
        import numpy as np
        import traceback
        import queue
        
        print(f"[Worker] Successfully imported torch and numpy")
        
        # Import vibevoice modules - use correct paths
        print(f"[Worker] Attempting to import vibevoice modules...")
        from vibevoice.runtime.model_loading import load_model_and_processor
        
        print(f"[Worker] Loading model {model_path} in child process (PID: {os.getpid()})")
        processor, model, _resolved = load_model_and_processor(
            model_path,
            model_settings,
            device=device,
            attn_implementation=attn_implementation,
        )
        
        # Setup scheduler
        model.model.noise_scheduler = model.model.noise_scheduler.from_config(
            model.model.noise_scheduler.config,
            algorithm_type='sde-dpmsolver++',
            beta_schedule='squaredcos_cap_v2'
        )
        model.set_ddpm_inference_steps(num_steps=inference_steps)
        
        print(f"[Worker] Model loaded successfully in child process")
        
        # Signal ready
        response_queue.put(("ready", None))
        
        # Process requests
        while True:
            try:
                request = request_queue.get(timeout=1.0)
                
                if request[0] == "shutdown":
                    print("[Worker] Shutdown requested")
                    break
                    
                elif request[0] == "generate":
                    # Unpack generation request
                    (_, text, voice_samples, cfg_scale, ddpm_steps, do_sample, 
                     temperature, top_p, top_k, negative_prompt, seed) = request
                    set_seed(seed)
                    
                    # Import AudioStreamer in worker
                    from vibevoice.modular.streamer import AudioStreamer
                    
                    # Set DDPM steps
                    if ddpm_steps is not None:
                        model.set_ddpm_inference_steps(num_steps=int(ddpm_steps))
                    
                    # Process inputs
                    inputs = processor(
                        text=[text],
                        voice_samples=[voice_samples],
                        padding=True,
                        return_tensors="pt",
                        return_attention_mask=True,
                    )
                    
                    # Prepare negative prompt
                    negative_ids = None
                    if negative_prompt and hasattr(processor, 'tokenizer'):
                        try:
                            negative_ids = processor.tokenizer(negative_prompt, return_tensors="pt").input_ids.to(model.device)
                        except Exception:
                            pass
                    
                    # Create audio streamer to catch EOS properly
                    audio_streamer = AudioStreamer(
                        batch_size=1,
                        stop_signal=None,
                        timeout=None
                    )
                    
                    print("[Worker] Starting generation with audio streamer...")
                    print("[Worker] 🧠 Generating speech tokens with autoregressive model...", flush=True)
                    
                    # Generate with streaming (to catch EOS)
                    with torch.no_grad():
                        outputs = model.generate(
                            **inputs,
                            max_new_tokens=None,
                            cfg_scale=cfg_scale,
                            tokenizer=processor.tokenizer,
                            generation_config={
                                'do_sample': bool(do_sample),
                                'temperature': float(temperature),
                                'top_p': float(top_p),
                                'top_k': int(top_k),
                            },
                            negative_prompt_ids=negative_ids,
                            audio_streamer=audio_streamer,  # Use streamer to catch EOS
                            verbose=False,
                            refresh_negative=True,
                        )
                    
                    # Collect all audio chunks from the streamer
                    print("[Worker] Collecting audio chunks from streamer...")
                    audio_stream = audio_streamer.get_stream(0)
                    audio_chunks = []
                    
                    for audio_chunk in audio_stream:
                        # Convert to numpy
                        if torch.is_tensor(audio_chunk):
                            if audio_chunk.dtype == torch.bfloat16:
                                audio_chunk = audio_chunk.float()
                            audio_np = audio_chunk.cpu().numpy().astype(np.float32)
                        else:
                            audio_np = np.array(audio_chunk, dtype=np.float32)
                        
                        # Ensure 1D
                        if len(audio_np.shape) > 1:
                            audio_np = audio_np.squeeze()
                        
                        audio_chunks.append(audio_np)
                    
                    # Concatenate all chunks
                    if audio_chunks:
                        audio_values = np.concatenate(audio_chunks)
                        print(f"[Worker] Generated {len(audio_chunks)} chunks, total duration: {len(audio_values)/24000:.2f}s")
                    else:
                        audio_values = np.array([], dtype=np.float32)
                        print("[Worker] Warning: No audio chunks generated")
                    
                    # Send result back
                    response_queue.put(("success", audio_values))
                    
            except queue.Empty:
                continue
            except Exception as e:
                print(f"[Worker] Error during generation: {e}")
                traceback.print_exc()
                response_queue.put(("error", str(e)))
                break
                
    except Exception as e:
        print(f"[Worker] Fatal error loading model: {e}")
        traceback.print_exc()
        response_queue.put(("error", str(e)))


class VibeVoiceDemo:
    def __init__(self, model_path: str, device: str = None, inference_steps: int = 5, debug: bool = False, load_on_demand: bool = False,
                 model_settings: ModelLoadingSettings | None = None):
        """Initialize the demo without loading model weights."""
        self.model_settings = model_settings or settings_from_args()
        self.model_path = model_path
        
        # Auto-detect device if not specified
        if device is None:
            self.device, device_name = detect_device()
            print(f"🔍 Auto-detected device: {device_name}")
        else:
            self.device = device
            if device == "cuda" and not torch.cuda.is_available():
                print("⚠️ CUDA requested but not available, falling back to CPU")
                self.device = "cpu"
            elif device == "mps" and not (hasattr(torch.backends, 'mps') and torch.backends.mps.is_available()):
                print("⚠️ MPS requested but not available, falling back to CPU")
                self.device = "cpu"
        self.inference_steps = inference_steps
        self.debug = debug
        self.load_on_demand = load_on_demand
        self.is_generating = False  # Track generation state
        self.stop_generation = False  # Flag to stop generation
        self.current_streamer = None  # Track current audio streamer
        self.model_loaded = False  # Track if model is loaded
        self.processor = None  # Will be loaded when needed
        self.model = None  # Will be loaded when needed
        
        # Multiprocessing for LOD mode (true VRAM cleanup)
        self.worker_process = None
        self.request_queue = None
        self.response_queue = None
        self.use_multiprocessing_lod = load_on_demand  # Use MP worker only in LOD mode

        self.available_models = {}
        for selection, repository in DEFAULT_MODEL_REPOSITORIES.items():
            self.available_models.setdefault(repository["folder"], selection)
        for name, path in discover_local_models(self.model_settings).items():
            self.available_models.setdefault(name, path)
        requested_path = Path(model_path).expanduser()
        root_path = requested_path if requested_path.is_absolute() else (Path(__file__).resolve().parent / requested_path)
        if requested_path.is_absolute() or root_path.is_dir():
            self.available_models[model_path] = str(root_path.resolve())
        elif "/" in model_path and model_path not in DEFAULT_MODEL_REPOSITORIES:
            self.available_models[model_path] = model_path

        self.model_path = normalize_model_selection(
            self.model_path,
            self.available_models,
        )

        # Build the page and voice choices before loading any model weights.
        self.setup_voice_presets()
        if load_on_demand:
            print("🔄 Load On Demand mode: Model will be loaded when first generation request is made")
        else:
            print("⏸️ No model loaded. Select a model and click Load Selected Model in the page.")
        
        # Removed legacy stop words storage from deprecated script system
        
    def ensure_model_loaded(self):
        """Ensure model is loaded, load it if not already loaded."""
        if not self.model_loaded:
            if self.use_multiprocessing_lod:
                self._spawn_worker_process()
            else:
                self.load_model()
            # Voice presets are already set up in __init__, no need to call again
    
    def _spawn_worker_process(self):
        """Spawn a worker process to load the model (LOD mode only)."""
        if self.model_loaded:
            return
            
        print(f"🔄 Spawning worker process for model {self.model_path}")
        
        # Create IPC queues
        self.request_queue = multiprocessing.Queue()
        self.response_queue = multiprocessing.Queue()
        
        attn_implementation = get_attention_implementation(self.device)
        
        # Resolve mapped path
        mapped_path = self.available_models.get(self.model_path, self.model_path)
        model_path_to_use = mapped_path
        
        # Spawn worker process
        self.worker_process = multiprocessing.Process(
            target=model_worker_process,
            args=(self.request_queue, self.response_queue, model_path_to_use, 
                  self.device, self.inference_steps, self.model_settings, attn_implementation)
        )
        self.worker_process.start()
        
        # Wait for worker to be ready
        print("⏳ Waiting for worker to load model...")
        try:
            status, data = self.response_queue.get(timeout=180)  # 3 minute timeout
            if status == "ready":
                print(f"✅ Worker process ready (PID: {self.worker_process.pid})")
                self.model_loaded = True
            elif status == "error":
                raise Exception(f"Worker failed to load model: {data}")
        except queue.Empty:
            raise Exception("Worker process timed out during model loading")

    def unload_model(self):
        """Unload the model to free VRAM."""
        # In LOD multiprocessing mode, kill the worker process
        if self.use_multiprocessing_lod and self.worker_process is not None:
            print(f"🔄 Terminating worker process (PID: {self.worker_process.pid}) to free VRAM")
            
            # Try graceful shutdown first
            try:
                self.request_queue.put(("shutdown", None), timeout=1.0)
                self.worker_process.join(timeout=5.0)
            except:
                pass
            
            # Force kill if still alive
            if self.worker_process.is_alive():
                print("⚠️ Force terminating worker process...")
                self.worker_process.terminate()
                self.worker_process.join(timeout=5.0)
                
                # Last resort: kill -9
                if self.worker_process.is_alive():
                    self.worker_process.kill()
                    self.worker_process.join()
            
            # Clean up
            self.worker_process = None
            if self.request_queue:
                self.request_queue.close()
                self.request_queue = None
            if self.response_queue:
                self.response_queue.close()
                self.response_queue = None
            
            self.model_loaded = False
            
            # Small delay to let OS reclaim GPU memory
            time.sleep(0.5)
            
            # Report VRAM after process kill
            if torch.cuda.is_available():
                allocated = torch.cuda.memory_allocated() / 1024**3
                reserved = torch.cuda.memory_reserved() / 1024**3
                print(f"✅ Worker terminated - VRAM: {allocated:.2f}GB allocated, {reserved:.2f}GB reserved")
            else:
                print(f"✅ Worker terminated and memory freed")
            
            return
        
        # Standard unloading for non-LOD mode
        if self.model_loaded and self.model is not None:
            print(f"Unloading model from {self.model_path} to free VRAM")
            
            # Move model to CPU first to release GPU memory
            if hasattr(self, 'model') and self.model is not None:
                try:
                    # Clear KV cache if model has it
                    if hasattr(self.model, 'model'):
                        if hasattr(self.model.model, 'language_model'):
                            # Clear language model cache
                            if hasattr(self.model.model.language_model, 'clear_cache'):
                                self.model.model.language_model.clear_cache()
                    
                    # Move all model components to CPU
                    self.model.to('cpu')
                    
                    # Clear the model's internal cache if it has one
                    if hasattr(self.model, 'clear_cache'):
                        self.model.clear_cache()
                        
                    # Explicitly set to eval and no_grad mode before deletion
                    self.model.eval()
                    
                except Exception as e:
                    print(f"Warning: Error moving model to CPU: {e}")
            
            # Clear current streamer first
            if hasattr(self, 'current_streamer') and self.current_streamer is not None:
                try:
                    self.current_streamer.end()
                except:
                    pass
                self.current_streamer = None
            
            # Delete model and processor references
            if hasattr(self, 'model'):
                del self.model
            if hasattr(self, 'processor'):
                del self.processor
                
            self.model = None
            self.processor = None
            self.model_loaded = False
            
            # Aggressive garbage collection
            import gc
            for _ in range(3):  # Multiple passes for circular references
                gc.collect()
            
            # Clear CUDA cache aggressively
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.synchronize()  # Wait for all CUDA operations to complete
                torch.cuda.ipc_collect()  # Collect IPC memory
                torch.cuda.empty_cache()  # Clear again after sync
                
                # Report VRAM usage after cleanup
                allocated = torch.cuda.memory_allocated() / 1024**3  # Convert to GB
                reserved = torch.cuda.memory_reserved() / 1024**3  # Convert to GB
                print(f"✅ Model unloaded - VRAM: {allocated:.2f}GB allocated, {reserved:.2f}GB reserved")
                
                # Reset peak memory stats
                torch.cuda.reset_peak_memory_stats()
            else:
                print(f"✅ Model unloaded and memory cleared")

    def switch_model(self, new_model_path: str):
        """Switch to a different model, unloading the current one if loaded."""
        if new_model_path == self.model_path and self.model_loaded:
            print(f"Model {new_model_path} is already loaded")
            return True

        print(f"Switching model from {self.model_path} to {new_model_path}")

        # Unload current model if loaded
        if self.model_loaded:
            self.unload_model()

        # Update model path and load new model
        self.model_path = new_model_path
        if self.load_on_demand:
            self.model_loaded = False
            print("🔄 Load On Demand: selected model will load on the next generation request")
            return True
        self.load_model()
        # Voice presets are already set up in __init__, no need to call again

        return True

    def load_model(self):
        """Load the selected VibeVoice model through the shared resolver."""
        print(f"Loading processor & model from {self.model_path}")
        attn_implementation = get_attention_implementation(self.device)
        print(f"🎯 Using attention implementation: {attn_implementation}")
        requested_path = self.available_models.get(self.model_path, self.model_path)
        self.processor, self.model, resolved = load_model_and_processor(
            requested_path,
            self.model_settings,
            device=self.device,
            attn_implementation=attn_implementation,
        )

        self.model.model.noise_scheduler = self.model.model.noise_scheduler.from_config(
            self.model.model.noise_scheduler.config,
            algorithm_type='sde-dpmsolver++',
            beta_schedule='squaredcos_cap_v2',
        )
        self.model.set_ddpm_inference_steps(num_steps=self.inference_steps)
        if hasattr(self.model.model, 'language_model'):
            print(f"Language model attention: {self.model.model.language_model.config._attn_implementation}")
        self.model_loaded = True
        print(f"✅ Model loaded successfully from {resolved.model_dir}")
    def setup_voice_presets(self):
        """Setup voice presets by scanning both demo voices and custom voices directories."""
        # Demo voices directory (relative to main.py)
        demo_voices_dir = os.path.join(os.path.dirname(__file__), "demo", "voices")
        # Custom voices directory
        custom_voices_dir = os.path.join(os.path.dirname(__file__), "custom_voices")
        
        self.voice_presets = {}
        
        # Scan demo voices directory
        if os.path.exists(demo_voices_dir):
            self._scan_voice_directory(demo_voices_dir, "", self.voice_presets)
            demo_count = len(self.voice_presets)
            print(f"Found {demo_count} demo voice files in {demo_voices_dir}")
        
        # Scan custom voices directory
        if os.path.exists(custom_voices_dir):
            custom_count_before = len(self.voice_presets)
            self._scan_voice_directory(custom_voices_dir, "custom_voices", self.voice_presets)
            custom_count_after = len(self.voice_presets)
            custom_added = custom_count_after - custom_count_before
            print(f"Found {custom_added} custom voice files in {custom_voices_dir}")
        
        # Sort the voice presets alphabetically by name (case-insensitive) for better UI
        self.voice_presets = dict(sorted(self.voice_presets.items(), key=lambda x: x[0].upper()))
        
        # Filter out voices that don't exist (this is now redundant but kept for safety)
        self.available_voices = {
            name: path for name, path in self.voice_presets.items()
            if os.path.exists(path)
        }
        
        if not self.available_voices:
            raise gr.Error("No voice presets found. Please add .wav files to the demo/voices or custom_voices directory.")
        
        print(f"Total available voices: {len(self.available_voices)}")
    
    def _scan_voice_directory(self, directory: str, prefix: str, voice_dict: dict):
        """Recursively scan a directory for voice files."""
        try:
            for item in os.listdir(directory):
                item_path = os.path.join(directory, item)
                
                if os.path.isfile(item_path):
                    # Check if it's an audio file
                    if item.lower().endswith(('.wav', '.mp3', '.flac', '.ogg', '.m4a', '.aac')):
                        # Remove extension to get the name
                        name = os.path.splitext(item)[0]
                        
                        # For custom voices, include the relative path in the display name
                        if prefix:
                            # Get relative path from custom_voices directory
                            rel_path = os.path.relpath(item_path, os.path.join(os.path.dirname(__file__), "custom_voices"))
                            display_name = f"{os.path.splitext(rel_path)[0]}"
                        else:
                            display_name = name
                        
                        voice_dict[display_name] = item_path
                
                elif os.path.isdir(item_path):
                    # Recursively scan subdirectories
                    self._scan_voice_directory(item_path, prefix, voice_dict)
                    
        except Exception as e:
            print(f"Error scanning directory {directory}: {e}")
    
    def read_audio(self, audio_path: str, target_sr: int = 24000) -> np.ndarray:
        """Read and preprocess audio file."""
        try:
            wav, sr = sf.read(audio_path)
            if len(wav.shape) > 1:
                wav = np.mean(wav, axis=1)
            if sr != target_sr:
                wav = librosa.resample(wav, orig_sr=sr, target_sr=target_sr)
            return wav
        except Exception as e:
            print(f"Error reading audio {audio_path}: {e}")
            return np.array([])
    
    def normalize_voice_samples(self, voice_samples: list) -> list:
        """Normalize all voice samples to similar RMS levels."""
        if not voice_samples:
            return voice_samples
        
        # Calculate RMS levels for each sample
        rms_levels = []
        for sample in voice_samples:
            if len(sample) > 0:
                rms = np.sqrt(np.mean(sample**2))
                rms_levels.append(rms)
            else:
                rms_levels.append(0)
        
        # Find the target RMS level (use the median to avoid outliers)
        valid_rms = [rms for rms in rms_levels if rms > 0]
        if not valid_rms:
            return voice_samples
        
        target_rms = np.median(valid_rms)
        
        # Normalize each sample to the target RMS level
        normalized_samples = []
        for i, sample in enumerate(voice_samples):
            if len(sample) > 0 and rms_levels[i] > 0:
                # Calculate gain factor
                gain_factor = target_rms / rms_levels[i]
                # Apply gain (with some headroom to prevent clipping)
                gain_factor = min(gain_factor, 3.0)  # Limit gain to 3x to prevent distortion
                normalized_sample = sample * gain_factor
                normalized_samples.append(normalized_sample)
            else:
                normalized_samples.append(sample)
        
        return normalized_samples
    
    def isolate_voice_samples(self, voice_samples: list, speaker_names: list = None, sample_rate: int = 24000) -> tuple:
        """
        Isolate vocals from voice samples to remove background music/noise.
        
        Uses a dedicated VocalIsolator instance that is cleaned up after processing
        to ensure VRAM is freed (especially important in LOD mode).
        
        Args:
            voice_samples: List of audio numpy arrays
            speaker_names: List of speaker names (for debug file saving)
            sample_rate: Sample rate of the audio samples
            
        Returns:
            Tuple of (isolated_samples list, error_message or None)
        """
        if not voice_samples:
            return voice_samples, None
        
        isolated_samples = []
        error_message = None
        isolator = None
        
        try:
            # Create a fresh isolator instance (will be cleaned up after)
            isolator = VocalIsolator(device=self.device, debug=self.debug, settings=self.model_settings)
            
            for i, sample in enumerate(voice_samples):
                if len(sample) > 0:
                    try:
                        # Use the vocal isolation module
                        isolated = isolator.isolate(sample, sample_rate=sample_rate)
                        isolated_samples.append(isolated)
                    except Exception as e:
                        # NO SILENT FALLBACK - fail openly and report the error
                        error_str = str(e)
                        if self.debug:
                            import traceback
                            error_str = f"{e}\n{traceback.format_exc()}"
                        error_message = f"Vocal isolation failed for sample {i}: {error_str}"
                        print(f"❌ {error_message}")
                        # Return original samples on failure
                        return voice_samples, error_message
                else:
                    isolated_samples.append(sample)
            
            # Debug mode: save isolated samples to custom_voices/debug/
            if self.debug and speaker_names:
                self._save_debug_voice_samples(isolated_samples, speaker_names, sample_rate, suffix="_isolated")
            
            return isolated_samples, None
            
        finally:
            # ALWAYS clean up the isolator to free VRAM
            if isolator is not None:
                del isolator
                # Also clear any global cache that might have been used
                clear_vocal_isolator_cache()
                # Force CUDA memory cleanup if available
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                if self.debug:
                    print("🔍 DEBUG: Vocal isolation model unloaded, VRAM freed")
    
    def _save_debug_voice_samples(self, samples: list, speaker_names: list, sample_rate: int, suffix: str = ""):
        """Save voice samples to debug directory for inspection."""
        try:
            debug_dir = os.path.join(os.path.dirname(__file__), "custom_voices", "debug")
            os.makedirs(debug_dir, exist_ok=True)
            
            for i, (sample, name) in enumerate(zip(samples, speaker_names)):
                if len(sample) > 0:
                    # Clean the speaker name for filename
                    clean_name = name.replace("/", "_").replace("\\", "_")
                    filename = f"{clean_name}{suffix}.wav"
                    filepath = os.path.join(debug_dir, filename)
                    
                    # Save as WAV file
                    sf.write(filepath, sample, sample_rate)
                    print(f"🔍 DEBUG: Saved voice sample to {filepath}")
        except Exception as e:
            print(f"🔍 DEBUG: Failed to save debug voice samples: {e}")

    def _save_generated_audio(self, audio_data: tuple, speaker_names: list) -> str:
        """Save generated audio with a date, speaker names, and unique counter."""
        try:
            from datetime import datetime

            output_dir = os.path.join(os.path.dirname(__file__), "output")
            os.makedirs(output_dir, exist_ok=True)
            sample_rate, audio_array = audio_data
            timestamp = datetime.now().strftime("%Y%m%d")

            clean_speakers = []
            for name in speaker_names[:4]:
                clean_name = name.split("/")[-1].split("\\")[-1]
                clean_name = clean_name.replace(" ", "-").replace("_", "-")
                if "-" in clean_name and len(clean_name.split("-")[0]) <= 2:
                    clean_name = "-".join(clean_name.split("-")[1:])
                clean_speakers.append(clean_name)

            speakers_str = "_".join(clean_speakers)
            counter = 1
            while True:
                filename = f"{timestamp}_{speakers_str}_audio-generation_{counter:03d}.wav"
                filepath = os.path.join(output_dir, filename)
                if not os.path.exists(filepath):
                    break
                counter += 1

            sf.write(filepath, audio_array, sample_rate)
            return filepath
        except Exception as e:
            print(f"❌ Failed to save output audio: {e}")
            traceback.print_exc()
            return None
    
    def generate_podcast_streaming(self, 
                                 num_speakers: int,
                                 script: str,
                                 speaker_1: str = None,
                                 speaker_2: str = None,
                                 speaker_3: str = None,
                                 speaker_4: str = None,
                                 cfg_scale: float = 1.6,
                                 diffusion_steps: int = None,
                                 do_sample: bool = True,
                                 temperature: float = 0.95,
                                 top_p: float = 0.95,
                                 top_k: int = 0,
                                 negative_prompt: str = "",
                                 isolate_voices: bool = True,
                                 normalize_voices: bool = False,
                                 seed: int = 42) -> Iterator[tuple]:
        try:
            
            # Reset stop flag and set generating state
            self.stop_generation = False
            self.is_generating = True
            
            # Validate inputs
            if not script.strip():
                self.is_generating = False
                raise gr.Error("Error: Please provide a script.")

            # Defend against common mistake
            script = script.replace("'", "'")
            
            if num_speakers < 1 or num_speakers > 4:
                self.is_generating = False
                raise gr.Error("Error: Number of speakers must be between 1 and 4.")
            seed = resolve_seed(seed)
            formatted_script = normalize_script(script, num_speakers)
            
            # Collect selected speakers
            selected_speakers = [speaker_1, speaker_2, speaker_3, speaker_4][:num_speakers]
            
            # Validate speaker selections
            for i, speaker in enumerate(selected_speakers):
                if not speaker or speaker not in self.available_voices:
                    self.is_generating = False
                    raise gr.Error(f"Error: Please select a valid speaker for Speaker {i+1}.")
            
            # Build initial log
            # Only set DDPM steps if not using multiprocessing (worker handles it)
            if not self.use_multiprocessing_lod:
                if diffusion_steps is not None and diffusion_steps != self.inference_steps:
                    self.model.set_ddpm_inference_steps(num_steps=int(diffusion_steps))
                else:
                    self.model.set_ddpm_inference_steps(num_steps=self.inference_steps)
            
            # Get effective diffusion steps for logging
            if self.use_multiprocessing_lod:
                effective_steps = diffusion_steps if diffusion_steps is not None else self.inference_steps
            else:
                effective_steps = self.model.ddpm_inference_steps

            log = f"🎙️ Generating audio with {num_speakers} speakers\n"
            log += f"📊 Parameters: CFG Scale={cfg_scale}, Diffusion Steps={effective_steps}, Sampling={do_sample}, Temp={temperature}, TopP={top_p}, TopK={top_k}\n"
            log += f"🎭 Speakers: {', '.join(selected_speakers)}\n"
            log += f"🎲 Seed: {seed}\n"
            
            # Check for stop signal
            if self.stop_generation:
                self.is_generating = False
                
                # Unload model if in LOD mode
                if self.load_on_demand and self.model_loaded:
                    self.unload_model()
                    print("🔄 Model unloaded to free VRAM after stopping generation")
                
                yield None, None, "🛑 Generation stopped by user", gr.update(visible=False)
                return
            
            # Load voice samples
            voice_samples = []
            for speaker_name in selected_speakers:
                audio_path = self.available_voices[speaker_name]
                audio_data = self.read_audio(audio_path)
                if len(audio_data) == 0:
                    self.is_generating = False
                    raise gr.Error(f"Error: Failed to load audio for {speaker_name}")
                voice_samples.append(audio_data)
            
            # Debug mode: save original samples before any processing
            if self.debug:
                self._save_debug_voice_samples(voice_samples, selected_speakers, 24000, suffix="_original")
            
            # Apply vocal isolation if requested (before normalization)
            if isolate_voices:
                voice_samples, isolation_error = self.isolate_voice_samples(
                    voice_samples, 
                    speaker_names=selected_speakers,
                    sample_rate=24000
                )
                if isolation_error:
                    log += f"⚠️ Vocal isolation FAILED: {isolation_error}\n"
                    log += "   Continuing with original voice samples...\n"
                else:
                    log += "🎤 Vocal isolation applied (model unloaded to free VRAM)\n"
            
            # Apply voice normalization if requested
            if normalize_voices:
                voice_samples = self.normalize_voice_samples(voice_samples)
                log += "🔊 Voice normalization applied\n"
                # Debug mode: save normalized samples
                if self.debug:
                    self._save_debug_voice_samples(voice_samples, selected_speakers, 24000, suffix="_normalized")
            
            # log += f"✅ Loaded {len(voice_samples)} voice samples\n"
            
            # Check for stop signal
            if self.stop_generation:
                self.is_generating = False
                
                # Unload model if in LOD mode
                if self.load_on_demand and self.model_loaded:
                    self.unload_model()
                    print("🔄 Model unloaded to free VRAM after stopping generation")
                
                yield None, None, "🛑 Generation stopped by user", gr.update(visible=False)
                return
            
            log += f"📝 Formatted script with {len(formatted_script.splitlines())} turns\n\n"
            log += "🔄 Processing with VibeVoice (streaming mode)...\n"
            
            # Check for stop signal before processing
            if self.stop_generation:
                self.is_generating = False
                
                # Unload model if in LOD mode
                if self.load_on_demand and self.model_loaded:
                    self.unload_model()
                    print("🔄 Model unloaded to free VRAM after stopping generation")
                
                yield None, None, "🛑 Generation stopped by user", gr.update(visible=False)
                return
            
            start_time = time.time()
            
            # ===== MULTIPROCESSING MODE (LOD with true VRAM cleanup) =====
            if self.use_multiprocessing_lod and self.worker_process:
                log += "🔄 Generating audio with worker process (no streaming, full VRAM cleanup)...\n"
                yield None, None, log, gr.update(visible=True)
                
                try:
                    # Generate using worker process
                    audio_values = self._generate_with_worker(
                        formatted_script, voice_samples, cfg_scale, diffusion_steps,
                        do_sample, temperature, top_p, top_k, negative_prompt, seed
                    )
                    
                    generation_time = time.time() - start_time
                    
                    # Ensure audio is 1D and properly normalized
                    if len(audio_values.shape) > 1:
                        audio_values = audio_values.squeeze()
                    
                    # Handle any NaN or infinity values
                    audio_values = np.nan_to_num(audio_values, nan=0.0, posinf=1.0, neginf=-1.0)
                    
                    # Convert to 16-bit for Gradio
                    audio_16bit = convert_to_16_bit_wav(audio_values)
                    
                    # Ensure it's actually int16 and contiguous in memory
                    if audio_16bit.dtype != np.int16:
                        audio_16bit = audio_16bit.astype(np.int16)
                    if not audio_16bit.flags['C_CONTIGUOUS']:
                        audio_16bit = np.ascontiguousarray(audio_16bit)
                    
                    sample_rate = 24000
                    audio_duration = len(audio_16bit) / sample_rate
                    
                    print(f"[Main] Audio shape: {audio_16bit.shape}, dtype: {audio_16bit.dtype}, duration: {audio_duration:.2f}s")
                    
                    final_log = log + f"⏱️ Generation completed in {generation_time:.2f} seconds\n"
                    final_log += f"🎵 Audio duration: {audio_duration:.2f} seconds\n"
                    final_log += "✨ Generation successful!\n"
                    final_log += "💡 Not satisfied? You can regenerate or adjust the CFG scale for different results."
                    
                    self.is_generating = False
                    
                    # Yield complete audio
                    yield None, (sample_rate, audio_16bit), final_log, gr.update(visible=False)
                    
                    # Unload (kill worker) to free VRAM
                    if self.load_on_demand and self.model_loaded:
                        self.unload_model()
                    
                    return
                    
                except Exception as e:
                    self.is_generating = False
                    error_msg = log + f"\n❌ Worker process error: {str(e)}"
                    print(error_msg)
                    traceback.print_exc()
                    
                    # Unload (kill worker) on error
                    if self.load_on_demand and self.model_loaded:
                        self.unload_model()
                    
                    yield None, None, error_msg, gr.update(visible=False)
                    return
            
            # ===== STANDARD MODE (with streaming) =====
            inputs = self.processor(
                text=[formatted_script],
                voice_samples=[voice_samples],
                padding=True,
                return_tensors="pt",
                return_attention_mask=True,
            )
            
            # Create audio streamer
            audio_streamer = AudioStreamer(
                batch_size=1,
                stop_signal=None,
                timeout=None
            )
            
            # Store current streamer for potential stopping
            self.current_streamer = audio_streamer
            
            # Start generation in a separate thread
            generation_thread = threading.Thread(
                target=self._generate_with_streamer,
                args=(inputs, cfg_scale, audio_streamer, do_sample, temperature, top_p, top_k, negative_prompt, seed)
            )
            generation_thread.start()
            
            # Wait for generation to actually start producing audio
            time.sleep(1)  # Reduced from 3 to 1 second

            # Check for stop signal after thread start
            if self.stop_generation:
                audio_streamer.end()
                generation_thread.join(timeout=5.0)  # Wait up to 5 seconds for thread to finish
                self.is_generating = False
                
                # Unload model if in LOD mode
                if self.load_on_demand and self.model_loaded:
                    self.unload_model()
                    print("🔄 Model unloaded to free VRAM after stopping generation")
                
                yield None, None, "🛑 Generation stopped by user", gr.update(visible=False)
                return

            # Collect audio chunks as they arrive
            sample_rate = 24000
            all_audio_chunks = []  # For final statistics
            pending_chunks = []  # Buffer for accumulating small chunks
            chunk_count = 0
            last_yield_time = time.time()
            min_yield_interval = 15 # Yield every 15 seconds
            min_chunk_size = sample_rate * 30 # At least 2 seconds of audio
            
            # Get the stream for the first (and only) sample
            audio_stream = audio_streamer.get_stream(0)
            
            has_yielded_audio = False
            has_received_chunks = False  # Track if we received any chunks at all
            
            for audio_chunk in audio_stream:
                # Check for stop signal in the streaming loop
                if self.stop_generation:
                    audio_streamer.end()
                    break
                    
                chunk_count += 1
                has_received_chunks = True  # Mark that we received at least one chunk
                
                # Convert tensor to numpy
                if torch.is_tensor(audio_chunk):
                    # Convert bfloat16 to float32 first, then to numpy
                    if audio_chunk.dtype == torch.bfloat16:
                        audio_chunk = audio_chunk.float()
                    audio_np = audio_chunk.cpu().numpy().astype(np.float32)
                else:
                    audio_np = np.array(audio_chunk, dtype=np.float32)
                
                # Ensure audio is 1D and properly normalized
                if len(audio_np.shape) > 1:
                    audio_np = audio_np.squeeze()
                
                # Convert to 16-bit for Gradio
                audio_16bit = convert_to_16_bit_wav(audio_np)
                
                # Store for final statistics
                all_audio_chunks.append(audio_16bit)
                
                # Add to pending chunks buffer
                pending_chunks.append(audio_16bit)
                
                # Calculate pending audio size
                pending_audio_size = sum(len(chunk) for chunk in pending_chunks)
                current_time = time.time()
                time_since_last_yield = current_time - last_yield_time
                
                # Decide whether to yield
                should_yield = False
                if not has_yielded_audio and pending_audio_size >= min_chunk_size:
                    # First yield: wait for minimum chunk size
                    should_yield = True
                    has_yielded_audio = True
                elif has_yielded_audio and (pending_audio_size >= min_chunk_size or time_since_last_yield >= min_yield_interval):
                    # Subsequent yields: either enough audio or enough time has passed
                    should_yield = True
                
                if should_yield and pending_chunks:
                    # Concatenate and yield only the new audio chunks
                    new_audio = np.concatenate(pending_chunks)
                    new_duration = len(new_audio) / sample_rate
                    total_duration = sum(len(chunk) for chunk in all_audio_chunks) / sample_rate
                    
                    log_update = log + f"🎵 Streaming: {total_duration:.1f}s generated (chunk {chunk_count})\n"
                    
                    # Yield streaming audio chunk and keep complete_audio as None during streaming
                    yield (sample_rate, new_audio), None, log_update, gr.update(visible=True)
                    
                    # Clear pending chunks after yielding
                    pending_chunks = []
                    last_yield_time = current_time
            
            # Yield any remaining chunks
            if pending_chunks:
                final_new_audio = np.concatenate(pending_chunks)
                total_duration = sum(len(chunk) for chunk in all_audio_chunks) / sample_rate
                log_update = log + f"🎵 Streaming final chunk: {total_duration:.1f}s total\n"
                yield (sample_rate, final_new_audio), None, log_update, gr.update(visible=True)
                has_yielded_audio = True  # Mark that we yielded audio
            
            # Wait for generation to complete (with timeout to prevent hanging)
            generation_thread.join(timeout=5.0)  # Increased timeout to 5 seconds

            # If thread is still alive after timeout, force end
            if generation_thread.is_alive():
                print("Warning: Generation thread did not complete within timeout")
                audio_streamer.end()
                generation_thread.join(timeout=5.0)

            # Clean up
            self.current_streamer = None
            self.is_generating = False
            
            generation_time = time.time() - start_time
            
            # Check if stopped by user
            if self.stop_generation:
                yield None, None, "🛑 Generation stopped by user", gr.update(visible=False)
                return
            
            # Debug logging
            # print(f"Debug: has_received_chunks={has_received_chunks}, chunk_count={chunk_count}, all_audio_chunks length={len(all_audio_chunks)}")
            
            # Check if we received any chunks but didn't yield audio
            if has_received_chunks and not has_yielded_audio and all_audio_chunks:
                # We have chunks but didn't meet the yield criteria, yield them now
                complete_audio = np.concatenate(all_audio_chunks)
                final_duration = len(complete_audio) / sample_rate
                
                final_log = log + f"⏱️ Generation completed in {generation_time:.2f} seconds\n"
                final_log += f"🎵 Final audio duration: {final_duration:.2f} seconds\n"
                final_log += f"📊 Total chunks: {chunk_count}\n"
                final_log += "✨ Generation successful! Complete audio is ready.\n"
                final_log += "💡 Not satisfied? You can regenerate or adjust the CFG scale for different results."
                
                # Yield the complete audio
                yield None, (sample_rate, complete_audio), final_log, gr.update(visible=False)
                
                # Unload model after successful generation if in LOD mode
                if self.load_on_demand and self.model_loaded:
                    self.unload_model()
                    print("🔄 Model unloaded to free VRAM after generation")
                return
            
            if not has_received_chunks:
                error_log = log + f"\n❌ Error: No audio chunks were received from the model. Generation time: {generation_time:.2f}s"
                
                # Unload model if in LOD mode
                if self.load_on_demand and self.model_loaded:
                    self.unload_model()
                    print("🔄 Model unloaded to free VRAM after error")
                
                yield None, None, error_log, gr.update(visible=False)
                return
            
            if not has_yielded_audio:
                error_log = log + f"\n❌ Error: Audio was generated but not streamed. Chunk count: {chunk_count}"
                
                # Unload model if in LOD mode
                if self.load_on_demand and self.model_loaded:
                    self.unload_model()
                    print("🔄 Model unloaded to free VRAM after error")
                
                yield None, None, error_log, gr.update(visible=False)
                return

            # Prepare the complete audio
            if all_audio_chunks:
                complete_audio = np.concatenate(all_audio_chunks)
                final_duration = len(complete_audio) / sample_rate
                
                final_log = log + f"⏱️ Generation completed in {generation_time:.2f} seconds\n"
                final_log += f"🎵 Final audio duration: {final_duration:.2f} seconds\n"
                final_log += f"📊 Total chunks: {chunk_count}\n"
                final_log += "✨ Generation successful! Complete audio is ready in the 'Complete Audio' tab.\n"
                final_log += "💡 Not satisfied? You can regenerate or adjust the CFG scale for different results."
                
                # Final yield: Clear streaming audio and provide complete audio
                yield None, (sample_rate, complete_audio), final_log, gr.update(visible=False)
                
                # Unload model after successful generation if in LOD mode
                if self.load_on_demand and self.model_loaded:
                    self.unload_model()
                    print("🔄 Model unloaded to free VRAM after generation")
            else:
                final_log = log + "❌ No audio was generated."
                
                # Unload model if in LOD mode
                if self.load_on_demand and self.model_loaded:
                    self.unload_model()
                    print("🔄 Model unloaded to free VRAM after failed generation")
                
                yield None, None, final_log, gr.update(visible=False)

        except gr.Error as e:
            # Handle Gradio-specific errors (like input validation)
            self.is_generating = False
            self.current_streamer = None
            error_msg = f"❌ Input Error: {str(e)}"
            print(error_msg)
            
            # Unload model if in LOD mode
            if self.load_on_demand and self.model_loaded:
                self.unload_model()
                print("🔄 Model unloaded to free VRAM after error")
            
            yield None, None, error_msg, gr.update(visible=False)
            
        except Exception as e:
            self.is_generating = False
            self.current_streamer = None
            error_msg = f"❌ An unexpected error occurred: {str(e)}"
            print(error_msg)
            traceback.print_exc()
            
            # Unload model if in LOD mode
            if self.load_on_demand and self.model_loaded:
                self.unload_model()
                print("🔄 Model unloaded to free VRAM after error")
            
            yield None, None, error_msg, gr.update(visible=False)
    
    def _generate_with_streamer(self, inputs, cfg_scale, audio_streamer, do_sample=True, temperature=0.95, top_p=0.95, top_k=0, negative_prompt: str = "", seed: int = 42):
        """Helper method to run generation with streamer in a separate thread."""
        try:
            # Check for stop signal before starting generation
            if self.stop_generation:
                audio_streamer.end()
                return
            set_seed(seed)
                
            # Define a stop check function that can be called from generate
            def check_stop_generation():
                return self.stop_generation
            
            # Print progress indicator for users
            print("🧠 Generating speech tokens with autoregressive model...", flush=True)
                
            # Prepare optional negative prompt ids
            negative_ids = None
            if negative_prompt and hasattr(self.processor, 'tokenizer'):
                try:
                    negative_ids = self.processor.tokenizer(negative_prompt, return_tensors="pt").input_ids.to(self.model.device)
                except Exception:
                    negative_ids = None

            outputs = self.model.generate(
                **inputs,
                max_new_tokens=None,
                cfg_scale=cfg_scale,
                tokenizer=self.processor.tokenizer,
                generation_config={
                    'do_sample': bool(do_sample),
                    'temperature': float(temperature),
                    'top_p': float(top_p),
                    'top_k': int(top_k),
                },
                negative_prompt_ids=negative_ids,
                audio_streamer=audio_streamer,
                stop_check_fn=check_stop_generation,  # Pass the stop check function
                verbose=self.debug,  # Enable verbose output in debug mode
                refresh_negative=True,
            )
            
        except Exception as e:
            print(f"Error in generation thread: {e}")
            traceback.print_exc()
            # Make sure to end the stream on error
            audio_streamer.end()
    
    def _generate_with_worker(self, formatted_script, voice_samples, cfg_scale, ddpm_steps,
                              do_sample, temperature, top_p, top_k, negative_prompt, seed=42):
        """
        Generate audio using the worker process (LOD multiprocessing mode).
        Returns complete audio (no streaming in this mode).
        """
        if not self.worker_process or not self.model_loaded:
            raise Exception("Worker process not available")
        
        print("[Main] Sending generation request to worker...")
        
        # Send request to worker
        request = ("generate", formatted_script, voice_samples, cfg_scale, ddpm_steps,
                   do_sample, temperature, top_p, top_k, negative_prompt, seed)
        self.request_queue.put(request)
        
        # Wait for response (with timeout)
        try:
            status, data = self.response_queue.get(timeout=600)  # 10 minute timeout
            
            if status == "success":
                print("[Main] Received audio from worker")
                return data  # numpy array of audio
            elif status == "error":
                raise Exception(f"Worker error: {data}")
            else:
                raise Exception(f"Unknown worker response: {status}")
                
        except queue.Empty:
            raise Exception("Worker timeout - generation took too long")
    
    def stop_audio_generation(self):
        """Stop the current audio generation process."""
        self.stop_generation = True
        
        # In multiprocessing mode, we can't really stop mid-generation
        # but we can kill the worker after
        if self.use_multiprocessing_lod:
            print("🛑 Stop requested - worker will be terminated after current generation")
        else:
            # Standard mode: stop the streamer
            if self.current_streamer is not None:
                try:
                    self.current_streamer.end()
                except Exception as e:
                    print(f"Error stopping streamer: {e}")
            print("🛑 Audio generation stop requested")
        
        # Unload model if in LOD mode (kills worker process or unloads model)
        if self.load_on_demand and self.model_loaded:
            self.unload_model()
            print("🔄 Model unloaded to free VRAM after stopping generation")
    
def create_demo_interface(demo_instance: VibeVoiceDemo):
    """Create the Gradio interface with streaming support."""

    def selected_choice(choices, preferred):
        """Keep a preferred option when present, otherwise use the first valid choice."""
        choices = [str(choice) for choice in choices]
        return preferred if preferred in choices else (choices[0] if choices else None)

    speaker_select_ids = [f"speaker-select-{i + 1}" for i in range(4)]
    model_select_id = "model-select"

    def render_speaker_select(index, choices, selected_value):
        speaker_label = f"Speaker {index + 1}"
        info = (
            f"Choose the voice used for {speaker_label}."
            if choices
            else "No voices are available. Add voices, then refresh the list."
        )
        return render_native_select(
            speaker_select_ids[index],
            speaker_label,
            choices,
            selected_choice(choices, selected_value),
            info=info,
            empty_message="No voices are available. Add voices, then refresh the list.",
        )

    def render_model_select(choices, selected_value):
        if choices:
            info = "Select a model (the current model unloads when switched)."
        else:
            info = "No models are available."
        return render_native_select(
            model_select_id,
            "Select Model",
            choices,
            selected_choice(choices, selected_value),
            info=info,
            empty_message="No models are available.",
        )
    
    theme_head = """
    <script>
    (() => {
        const storageKey = "vibevoice-theme";
        let savedTheme;
        try { savedTheme = localStorage.getItem(storageKey); } catch {}
        const initialTheme = savedTheme === "light" ? "light" : "dark";
        const url = new URL(window.location.href);
        // An explicit mode prevents Gradio from subscribing to system changes.
        if (url.searchParams.get("__theme") !== initialTheme) {
            url.searchParams.set("__theme", initialTheme);
            window.location.replace(url);
            return;
        }

        window.vibevoiceTheme = {
            apply(theme) {
                if (theme !== "light" && theme !== "dark") {
                    throw new Error("Unknown theme: " + theme);
                }
                document.body.classList.toggle("dark", theme === "dark");
                const url = new URL(window.location.href);
                url.searchParams.set("__theme", theme);
                window.history.replaceState(null, "", url);
                try { localStorage.setItem(storageKey, theme); } catch {}
                return theme === "dark" ? "☀️ Light mode" : "🌙 Dark mode";
            },
            toggle() {
                return this.apply(document.body.classList.contains("dark") ? "light" : "dark");
            }
        };
        window.vibevoiceTheme.apply(initialTheme);
    })();
    </script>
    """

    custom_css = """
    .gradio-container {
        background: var(--body-background-fill);
        font-family: 'SF Pro Display', -apple-system, BlinkMacSystemFont, sans-serif;
        color: var(--body-text-color);
        color-scheme: light;
    }

    .dark .gradio-container {
        color-scheme: dark;
    }

    #theme-toggle {
        max-width: 180px;
        margin-left: auto;
    }
    
    /* Header styling */
    .main-header {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 20px;
        margin-bottom: 2rem;
        text-align: center;
        box-shadow: 0 10px 40px rgba(102, 126, 234, 0.3);
    }
    
    .main-header h1 {
        color: white;
        font-size: 2.5rem;
        font-weight: 700;
        margin: 0;
        text-shadow: 0 2px 4px rgba(0,0,0,0.3);
    }
    
    .main-header p {
        color: rgba(255,255,255,0.9);
        font-size: 1.1rem;
        margin: 0.5rem 0 0 0;
    }
    
    /* Card styling */
    .settings-card, .generation-card {
        background: var(--background-fill-secondary);
        backdrop-filter: blur(10px);
        border: 1px solid var(--border-color-primary);
        border-radius: 16px;
        padding: 1.5rem;
        margin-bottom: 1rem;
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.3);
        color: var(--body-text-color);
    }
    
        /* Speaker selection styling */
    .speaker-grid {
        display: grid;
        gap: 1rem;
        margin-bottom: 1rem;
    }

    .speaker-item {
        background: linear-gradient(135deg, var(--background-fill-primary), var(--background-fill-secondary));
        border: 1px solid var(--border-color-primary);
        border-radius: 12px;
        padding: 1rem;
        color: var(--body-text-color);
        font-weight: 500;
    }

    .native-select-widget {
        display: flex;
        flex-direction: column;
        gap: 0.45rem;
        width: 100%;
    }

    .native-select-label {
        color: var(--body-text-color);
        font-size: 0.95rem;
        font-weight: 600;
    }

    select.native-select {
        appearance: auto;
        width: 100%;
        min-height: 2.8rem;
        padding: 0.65rem 0.85rem;
        border: 1px solid var(--border-color-primary);
        border-radius: 10px;
        background: var(--input-background-fill);
        color: var(--body-text-color);
        color-scheme: inherit;
        font: inherit;
        cursor: pointer;
    }

    select.native-select option {
        background-color: var(--input-background-fill);
        color: var(--body-text-color);
    }

    select.native-select:focus-visible {
        outline: 3px solid rgba(129, 140, 248, 0.95);
        outline-offset: 2px;
        border-color: #a5b4fc;
    }

    select.native-select:disabled {
        color: var(--body-text-color-subdued);
        cursor: not-allowed;
        opacity: 0.85;
    }

    .native-select-help {
        margin: 0;
        color: var(--body-text-color-subdued);
        font-size: 0.82rem;
        line-height: 1.35;
    }

    .model-select-field {
        margin-bottom: 0.5rem;
    }
    
    /* Streaming indicator */
    .streaming-indicator {
        display: inline-block;
        width: 10px;
        height: 10px;
        background: #22c55e;
        border-radius: 50%;
        margin-right: 8px;
        animation: pulse 1.5s infinite;
    }
    
    @keyframes pulse {
        0% { opacity: 1; transform: scale(1); }
        50% { opacity: 0.5; transform: scale(1.1); }
        100% { opacity: 1; transform: scale(1); }
    }
    
    /* Queue status styling */
    .queue-status {
        background: var(--background-fill-secondary);
        border: 1px solid rgba(14, 165, 233, 0.4);
        border-radius: 8px;
        padding: 0.75rem;
        margin: 0.5rem 0;
        text-align: center;
        font-size: 0.9rem;
        color: var(--color-accent);
    }
    
    .generate-btn {
        background: linear-gradient(135deg, #059669 0%, #0d9488 100%);
        border: none;
        border-radius: 12px;
        padding: 1rem 2rem;
        color: white;
        font-weight: 600;
        font-size: 1.1rem;
        box-shadow: 0 4px 20px rgba(5, 150, 105, 0.4);
        transition: all 0.3s ease;
    }
    
    .generate-btn:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 25px rgba(5, 150, 105, 0.6);
    }
    
    .stop-btn {
        background: linear-gradient(135deg, #ef4444 0%, #dc2626 100%);
        border: none;
        border-radius: 12px;
        padding: 1rem 2rem;
        color: white;
        font-weight: 600;
        font-size: 1.1rem;
        box-shadow: 0 4px 20px rgba(239, 68, 68, 0.4);
        transition: all 0.3s ease;
    }
    
    .stop-btn:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 25px rgba(239, 68, 68, 0.6);
    }
    
        /* Audio player styling */
    .audio-output {
        background: linear-gradient(135deg, var(--background-fill-primary), var(--background-fill-secondary));
        border-radius: 16px;
        padding: 1.5rem;
        border: 1px solid var(--border-color-primary);
        color: var(--body-text-color);
    }

    .complete-audio-section {
        margin-top: 1rem;
        padding: 1rem;
        background: linear-gradient(135deg, #dcfce7 0%, #bbf7d0 100%);
        border: 1px solid rgba(34, 197, 94, 0.4);
        border-radius: 12px;
        color: #166534;
    }

    .dark .complete-audio-section {
        background: linear-gradient(135deg, #064e3b 0%, #065f46 100%);
        color: #d1fae5;
    }
    
        /* Text areas */
    .script-input, .log-output {
        background: var(--input-background-fill) !important;
        border: 1px solid var(--border-color-primary) !important;
        border-radius: 12px !important;
        color: var(--body-text-color) !important;
        font-family: 'JetBrains Mono', monospace !important;
    }

    .script-input textarea::placeholder, .log-output textarea::placeholder {
        color: var(--body-text-color-subdued) !important;
    }
    
        /* Sliders */
    .slider-container {
        background: var(--background-fill-primary);
        border: 1px solid var(--border-color-primary);
        border-radius: 8px;
        padding: 1rem;
        margin: 0.5rem 0;
        color: var(--body-text-color);
    }

    /* Labels and text */
    .gradio-container label {
        color: var(--body-text-color) !important;
        font-weight: 600 !important;
    }

    .gradio-container .markdown {
        color: var(--body-text-color) !important;
    }
    
    /* Responsive design */
    @media (max-width: 768px) {
        .main-header h1 { font-size: 2rem; }
        .settings-card, .generation-card { padding: 1rem; }
    }
    
    /* Dropdown improvements */
    .gradio-container .dropdown {
        max-height: 200px !important;
        overflow-y: auto !important;
        scrollbar-width: thin !important;
        scrollbar-color: var(--border-color-primary) var(--background-fill-primary) !important;
    }
    
    .gradio-container .dropdown::-webkit-scrollbar {
        width: 8px !important;
    }
    
    .gradio-container .dropdown::-webkit-scrollbar-track {
        background: var(--background-fill-primary) !important;
        border-radius: 4px !important;
    }
    
    .gradio-container .dropdown::-webkit-scrollbar-thumb {
        background: var(--border-color-primary) !important;
        border-radius: 4px !important;
    }
    
    .gradio-container .dropdown::-webkit-scrollbar-thumb:hover {
        background: var(--border-color-secondary) !important;
    }
    
    /* Prevent dropdown from causing page scroll */
    .gradio-container .dropdown-panel {
        position: fixed !important;
        z-index: 1000 !important;
        max-height: 300px !important;
        overflow-y: auto !important;
    }


    """
    
    with gr.Blocks(
        title="VibeVoice - Speech Generation and Transcription",
        css=custom_css,
        head=theme_head,
        theme=gr.themes.Soft(
            primary_hue="blue",
            secondary_hue="purple",
            neutral_hue="slate",
        ).set(
            body_background_fill="linear-gradient(135deg, #f8fafc 0%, #e2e8f0 100%)",
            body_background_fill_dark="linear-gradient(135deg, #0f172a 0%, #1e293b 100%)",
            background_fill_primary="#f1f5f9",
            background_fill_primary_dark="#1e293b",
            background_fill_secondary="#ffffff",
            background_fill_secondary_dark="#0f172a",
            block_background_fill="*background_fill_primary",
            block_background_fill_dark="*background_fill_primary",
            block_label_background_fill="*background_fill_secondary",
            block_label_background_fill_dark="*background_fill_secondary",
            input_background_fill="*background_fill_secondary",
            input_background_fill_dark="*background_fill_secondary",
            button_secondary_background_fill="*background_fill_primary",
            button_secondary_background_fill_dark="*background_fill_primary",
            button_secondary_background_fill_hover="#e2e8f0",
            button_secondary_background_fill_hover_dark="#334155",
            button_secondary_text_color="*body_text_color",
            button_secondary_text_color_dark="*body_text_color",
            border_color_primary="#cbd5e1",
            border_color_primary_dark="#334155",
            color_accent_soft="#667eea",
            body_text_color="#1e293b",
            body_text_color_dark="#e2e8f0",
            body_text_color_subdued="#64748b",
            body_text_color_subdued_dark="#94a3b8",
        )
    ) as interface:
        theme_toggle = gr.Button("☀️ Light mode", elem_id="theme-toggle", size="sm")
        interface.load(
            fn=None,
            inputs=[],
            outputs=theme_toggle,
            js="() => window.vibevoiceTheme.apply(document.body.classList.contains('dark') ? 'dark' : 'light')",
            queue=False,
        )
        theme_toggle.click(
            fn=None,
            inputs=[],
            outputs=theme_toggle,
            js="() => window.vibevoiceTheme.toggle()",
            queue=False,
        )

        # Header
        gr.HTML("""
        <div class="main-header">
            <h1>🎙️ VibeVoice</h1>
            <p>Generate speech and transcribe audio with VibeVoice</p>
        </div>
        """)
        
        with gr.Tabs():
            with gr.Tab("TTS"):
                with gr.Row():
                    # Left column - Settings
                    with gr.Column(scale=1, elem_classes="settings-card"):
                        gr.Markdown("### 🎛️ **Audio Settings**")

                        # Number of speakers
                        num_speakers = gr.Slider(
                            minimum=1,
                            maximum=4,
                            value=2,
                            step=1,
                            label="Number of Speakers",
                            elem_classes="slider-container"
                        )

                        # Speaker selection
                        gr.Markdown("### 🎭 **Speaker Selection**")

                        available_speaker_names = list(demo_instance.available_voices.keys())
                        default_speakers = ['en-Alice_woman', 'en-Carter_man', 'en-Frank_man', 'en-Maya_woman']

                        speaker_selections = []
                        for i in range(4):
                            default_value = default_speakers[i] if i < len(default_speakers) else None
                            speaker_label = f"Speaker {i + 1}"
                            speaker = gr.HTML(
                                value=render_speaker_select(i, available_speaker_names, default_value),
                                label=speaker_label,
                                show_label=False,
                                visible=(i < 2) if i < 2 else "hidden",  # Keep hidden selects in the DOM for value retention.
                                elem_classes="speaker-item",
                                elem_id=f"speaker-select-field-{i + 1}",
                                min_height=0,
                                padding=False,
                            )
                            speaker_selections.append(speaker)
                        # Refresh voices button
                        refresh_voices_btn = gr.Button(
                            "🔄 Refresh Voices",
                            variant="secondary"
                        )

                        # Voice Input Settings
                        with gr.Accordion("🎤 Voice Input Settings", open=False):
                            isolate_voices = gr.Checkbox(
                                value=True,
                                label="Isolate input voices",
                                info="Remove background music/noise from voice samples using AI vocal isolation (recommended)"
                            )
                            normalize_voices = gr.Checkbox(
                                value=False,
                                label="Normalize voices",
                                info="Normalize all voice samples to similar volume levels to prevent jarring volume differences"
                            )

                        # Output Settings
                        with gr.Accordion("💾 Output Settings", open=False):
                            save_output = gr.Checkbox(
                                value=True,
                                label="Save generated audio to output folder",
                                info="Automatically save generated audio to output/ directory with timestamp and speaker names"
                            )

                        # Model selector
                        gr.Markdown("### 🤖 **Model Selection**")
                        model_choices = list(demo_instance.available_models.keys())
                        selected_model = demo_instance.model_path if demo_instance.model_path in demo_instance.available_models else (model_choices[0] if model_choices else None)
                        model_selector = gr.HTML(
                            value=render_model_select(model_choices, selected_model),
                            label="Select Model",
                            show_label=False,
                            elem_id="model-select-field",
                            elem_classes="model-select-field",
                            min_height=0,
                            padding=False,
                        )

                        load_model_btn = gr.Button(
                            "🔄 Load Selected Model",
                            variant="secondary",
                            elem_classes="model-btn"
                        )

                        # Advanced settings
                        gr.Markdown("### ⚙️ **Advanced Settings**")

                        # Sampling parameters (contains all generation settings)
                        with gr.Accordion("Generation Parameters", open=False):
                            seed_input = gr.Number(
                                value=42, precision=0, minimum=0, maximum=4294967295,
                                label="Seed", info="Use the same positive seed to repeat a run; 0 chooses a random seed shown in the log.",
                            )
                            cfg_scale = gr.Slider(
                                minimum=1.0,
                                maximum=2.0,
                                value=1.6,
                                step=0.05,
                                label="CFG Scale (Guidance Strength)",
                                # info="Higher values increase adherence to text",
                                elem_classes="slider-container"
                            )
                            ddpm_steps = gr.Slider(
                                minimum=5,
                                maximum=30,
                                value=demo_instance.inference_steps,
                                step=1,
                                label="Diffusion Steps (quality vs speed)",
                                elem_classes="slider-container"
                            )
                            do_sample = gr.Checkbox(
                                value=True,
                                label="Enable sampling (adds variability)",
                            )
                            temperature = gr.Slider(
                                minimum=0.1,
                                maximum=1.5,
                                value=0.95,
                                step=0.05,
                                label="Temperature",
                                elem_classes="slider-container"
                            )
                            top_p = gr.Slider(
                                minimum=0.0,
                                maximum=1.0,
                                value=0.95,
                                step=0.01,
                                label="Top-p",
                                elem_classes="slider-container"
                            )
                            top_k = gr.Slider(
                                minimum=0,
                                maximum=100,
                                value=0,
                                step=1,
                                label="Top-k",
                                elem_classes="slider-container"
                            )
                            negative_prompt = gr.Textbox(
                                label="Negative Prompt (optional)",
                                placeholder="Words or patterns to avoid...",
                                lines=2,
                                max_lines=4,
                                value=""
                            )

                    # Right column - Generation
                    with gr.Column(scale=2, elem_classes="generation-card"):
                        gr.Markdown("### 📝 **Script Input**")

                        script_input = gr.Textbox(
                            label="Conversation Script",
                            placeholder="""Enter your dialogue script here. You can format it as:

        [1] Welcome to our conversation today!
        [2] Thanks for having me. I'm excited to discuss...

        Speaker 1: Text is also supported. Unlabelled text after a marker continues that turn.
        Or paste plain text directly and it will auto-assign speakers.""",
                            lines=18,
                            max_lines=40,
                            elem_classes="script-input"
                        )

                        # Generate Audio Button (full width)
                        generate_btn = gr.Button(
                            "🚀 Generate Audio",
                            size="lg",
                            variant="primary",
                            elem_classes="generate-btn"
                        )

                        # Stop button
                        stop_btn = gr.Button(
                            "🛑 Stop Generation",
                            size="lg",
                            variant="stop",
                            elem_classes="stop-btn",
                            visible=False
                        )

                        # Streaming status indicator
                        streaming_status = gr.HTML(
                            value="""
                            <div style="background: linear-gradient(135deg, #dcfce7 0%, #bbf7d0 100%);
                                        border: 1px solid rgba(34, 197, 94, 0.3);
                                        border-radius: 8px;
                                        padding: 0.75rem;
                                        margin: 0.5rem 0;
                                        text-align: center;
                                        font-size: 0.9rem;
                                        color: #166534;">
                                <span class="streaming-indicator"></span>
                                <strong>LIVE STREAMING</strong> - Audio is being generated in real-time
                            </div>
                            """,
                            visible=False,
                            elem_id="streaming-status"
                        )

                        # Output section
                        gr.Markdown("### 🎵 **Generated Audio**")

                        # Streaming audio output (outside of tabs for simpler handling)
                        # Build kwargs conditionally based on Gradio version
                        streaming_audio_kwargs = {
                            "label": "Streaming Audio (Real-time)",
                            "type": "numpy",
                            "elem_classes": "audio-output",
                            "streaming": True,
                            "autoplay": True,
                            "visible": True
                        }
                        if GRADIO_HAS_SHOW_DOWNLOAD:
                            streaming_audio_kwargs["show_download_button"] = False

                        audio_output = gr.Audio(**streaming_audio_kwargs)

                        # Complete audio output (non-streaming)
                        complete_audio_kwargs = {
                            "label": "Complete Audio (Download after generation)",
                            "type": "numpy",
                            "elem_classes": "audio-output complete-audio-section",
                            "streaming": False,
                            "autoplay": False,
                            "visible": False,
                            "elem_id": "complete-audio-output"
                        }
                        if GRADIO_HAS_SHOW_DOWNLOAD:
                            complete_audio_kwargs["show_download_button"] = True

                        complete_audio_output = gr.Audio(**complete_audio_kwargs)

                        # Simple gain control for the audio player
                        with gr.Row():
                            gain_control = gr.Slider(
                                minimum=-20.0,
                                maximum=20.0,
                                value=0.0,
                                step=0.1,
                                label="Gain (dB)",
                                elem_id="gain-control",
                                interactive=True
                            )

                            gain_reset_btn = gr.Button(
                                value="Reset",
                                variant="secondary",
                                size="sm",
                                elem_id="gain-reset-btn"
                            )

                        gr.Markdown("""
                        *💡 **Streaming**: Audio plays as it's being generated (may have slight pauses)<br>
                        *💡 **Complete Audio**: Will appear below after generation finishes*
                        """)

                        # Generation log
                        log_output = gr.Textbox(
                            label="Generation Log",
                            value=("" if demo_instance.model_loaded else
                                   "No model loaded. Select a model and click Load Selected Model when ready."),
                            lines=8,
                            max_lines=15,
                            interactive=False,
                            elem_classes="log-output"
                        )

                def update_speaker_visibility(num_speakers, *selected_speakers):
                    current_choices = list(demo_instance.available_voices.keys())
                    return [
                        gr.update(
                            value=render_speaker_select(
                                index,
                                current_choices,
                                selected_speakers[index] if index < len(selected_speakers) else None,
                            ),
                            visible=(index < int(num_speakers)) if index < int(num_speakers) else "hidden",
                        )
                        for index in range(4)
                    ]

                # Refresh the list of voices from disk and update the native selects.
                def refresh_voices(*selected_speakers):
                    demo_instance.setup_voice_presets()
                    new_choices = list(demo_instance.available_voices.keys())
                    return [
                        render_speaker_select(
                            index,
                            new_choices,
                            selected_speakers[index] if index < len(selected_speakers) else None,
                        )
                        for index in range(4)
                    ]

                num_speakers.change(
                    fn=update_speaker_visibility,
                    inputs=[num_speakers] + speaker_selections,
                    outputs=speaker_selections,
                    js=(
                        f"(...values) => {{ {native_select_reader_js(speaker_select_ids, [1, 2, 3, 4])} "
                        "return [values[0], ...selected]; }"
                    ),
                )

                # Wire refresh button to update dropdown choices
                refresh_voices_btn.click(
                    fn=refresh_voices,
                    inputs=speaker_selections,
                    outputs=speaker_selections,
                    js=select_values_js(speaker_select_ids, [0, 1, 2, 3]),
                    queue=False
                )

            with gr.Tab("Realtime TTS"):
                realtime_controller = build_realtime_controls(demo_instance, interface)
            with gr.Tab("ASR"):
                asr_models = discover_asr_models(demo_instance.model_settings)
                asr_model = gr.Dropdown(choices=list(asr_models), value=next(iter(asr_models), None),
                                        label="ASR Model", interactive=True)
                asr_upload = gr.Audio(sources=["upload"], type="filepath", label="Upload Audio")
                asr_context = gr.Textbox(label="Context / Hotwords (optional)", placeholder="Names or terms that appear in the audio")
                transcribe_btn = gr.Button("Transcribe", variant="primary")
                asr_transcript = gr.Textbox(label="Transcript", lines=8, interactive=False)
                asr_segments = gr.JSON(label="Segments (timestamps and speaker IDs)")
                asr_status = gr.Textbox(label="Transcription Status", interactive=False,
                                        value="Ready" if asr_models else "No local VibeVoice-ASR-HF checkpoint found in the model root.")

        def transcribe_upload(audio_path, selected_model, context):
            try:
                if not audio_path:
                    raise ValueError("Upload an audio file before transcribing.")
                if selected_model not in asr_models:
                    raise ValueError("Select a local ASR model before transcribing.")
                asr_python()  # Check the runtime before releasing the current TTS model.
                yield "", [], "Loading ASR and transcribing…"
                realtime_controller.unload()
                if demo_instance.model_loaded:
                    demo_instance.unload_model()
                payload = run_transcription(audio_path, asr_models[selected_model], demo_instance.device, context)
                yield payload["transcript"], payload["segments"], payload.get("warning") or "Transcription complete."
            except Exception as exc:
                yield "", [], f"Transcription failed: {exc}"

        transcribe_btn.click(transcribe_upload, inputs=[asr_upload, asr_model, asr_context],
                             outputs=[asr_transcript, asr_segments, asr_status],
                             concurrency_id="model_operations", concurrency_limit=1)

        def generate_podcast_wrapper(num_speakers, script, *speakers_and_params):
            """Wrapper function to handle the streaming generation call."""
            try:
                # Load when generating if needed (including after ASR unloads TTS).
                realtime_controller.unload()
                demo_instance.ensure_model_loaded()

                # Extract speakers and parameters
                speakers = speakers_and_params[:4]  # First 4 are speaker selections
                cfg_scale = speakers_and_params[4]   # CFG scale
                ddpm_steps_val = int(speakers_and_params[5])
                do_sample_val = bool(speakers_and_params[6])
                temperature_val = float(speakers_and_params[7])
                top_p_val = float(speakers_and_params[8])
                top_k_val = int(speakers_and_params[9])
                negative_prompt_val = speakers_and_params[10] or ""
                isolate_voices_val = bool(speakers_and_params[11]) if len(speakers_and_params) > 11 else True
                normalize_voices_val = bool(speakers_and_params[12]) if len(speakers_and_params) > 12 else False
                save_output_val = bool(speakers_and_params[13]) if len(speakers_and_params) > 13 else True
                seed_val = resolve_seed(speakers_and_params[14]) if len(speakers_and_params) > 14 else 42

                # Clear audio outputs and reset visibility at start
                yield None, gr.update(value=None, visible=False), "🎙️ Starting generation...", gr.update(visible=True), gr.update(visible=False), gr.update(visible=True)
                
                # The generator will yield multiple times
                final_log = "Starting generation..."
                
                for streaming_audio, complete_audio, log, streaming_visible in demo_instance.generate_podcast_streaming(
                    num_speakers=int(num_speakers),
                    script=script,
                    speaker_1=speakers[0],
                    speaker_2=speakers[1],
                    speaker_3=speakers[2],
                    speaker_4=speakers[3],
                    cfg_scale=cfg_scale,
                    diffusion_steps=ddpm_steps_val,
                    do_sample=do_sample_val,
                    temperature=temperature_val,
                    top_p=top_p_val,
                    top_k=top_k_val,
                    negative_prompt=negative_prompt_val,
                    isolate_voices=isolate_voices_val,
                    normalize_voices=normalize_voices_val,
                    seed=seed_val
                ):
                    final_log = log
                    
                    # Check if we have complete audio (final yield)
                    if complete_audio is not None:
                        # Final state: clear streaming, show complete audio
                        # Save output file if requested
                        if save_output_val:
                            # Get active speaker names (only up to num_speakers)
                            active_speakers = [speakers[i] for i in range(int(num_speakers))]
                            saved_path = demo_instance._save_generated_audio(complete_audio, active_speakers)
                            if saved_path:
                                log = log + f"\n💾 Audio saved to: {saved_path}\n"
                        
                        yield None, gr.update(value=complete_audio, visible=True), log, gr.update(visible=False), gr.update(visible=True), gr.update(visible=False)
                        
                        # Cache the original audio for gain processing
                        cache_original_audio(complete_audio)
                    else:
                        # Streaming state: update streaming audio only
                        if streaming_audio is not None:
                            yield streaming_audio, gr.update(visible=False), log, streaming_visible, gr.update(visible=False), gr.update(visible=True)
                        else:
                            # No new audio, just update status
                            yield None, gr.update(visible=False), log, streaming_visible, gr.update(visible=False), gr.update(visible=True)

                # Unload model after successful generation if in LOD mode
                # Note: Model unloading is now handled in the generation method itself
                # to ensure it happens after the final audio yield

            except Exception as e:
                error_msg = f"❌ A critical error occurred in the wrapper: {str(e)}"
                print(error_msg)
                traceback.print_exc()
                
                # Unload model if in LOD mode
                if demo_instance.load_on_demand and demo_instance.model_loaded:
                    demo_instance.unload_model()
                    print("🔄 Model unloaded to free VRAM after wrapper error")
                
                # Reset button states on error
                yield None, gr.update(value=None, visible=False), error_msg, gr.update(visible=False), gr.update(visible=True), gr.update(visible=False)
        
        def stop_generation_handler():
            """Handle stopping generation."""
            demo_instance.stop_audio_generation()
            # Return values for: log_output, streaming_status, generate_btn, stop_btn
            return "🛑 Generation stopped.", gr.update(visible=False), gr.update(visible=True), gr.update(visible=False)
        
        # Add a clear audio function
        def clear_audio_outputs():
            """Clear both audio outputs before starting a new generation."""
            return None, gr.update(value=None, visible=False)

        # Connect generation button with streaming outputs
        generate_btn.click(
            fn=clear_audio_outputs,
            inputs=[],
            outputs=[audio_output, complete_audio_output],
            queue=False
        ).then(
            fn=generate_podcast_wrapper,
            inputs=[num_speakers, script_input] + speaker_selections + [cfg_scale, ddpm_steps, do_sample, temperature, top_p, top_k, negative_prompt, isolate_voices, normalize_voices, save_output, seed_input],
            outputs=[audio_output, complete_audio_output, log_output, streaming_status, generate_btn, stop_btn],
            js=(
                f"(...values) => {{ {native_select_reader_js(speaker_select_ids, [2, 3, 4, 5])} "
                "return [...values.slice(0, 2), ...selected, ...values.slice(6)]; }"
            ),
            queue=True, concurrency_id="model_operations", concurrency_limit=1
        )
        
        # Connect stop button
        stop_btn.click(
            fn=stop_generation_handler,
            inputs=[],
            outputs=[log_output, streaming_status, generate_btn, stop_btn],
            queue=False  # Don't queue stop requests
        ).then(
            # Clear both audio outputs after stopping
            fn=lambda: (None, None),
            inputs=[],
            outputs=[audio_output, complete_audio_output],
            queue=False
        )

        # Model switching function
        def switch_model(selected_model, *selected_speakers):
            """Switch to the selected model."""
            model_choices = list(demo_instance.available_models.keys())
            previous_model = demo_instance.model_path

            if selected_model not in demo_instance.available_models:
                status_msg = f"❌ Unknown model selection: {selected_model}"
                return (status_msg, render_model_select(model_choices, previous_model)) + tuple(
                    gr.update() for _ in range(4)
                )

            try:
                realtime_controller.unload()
                success = demo_instance.switch_model(selected_model)
                if success:
                    # Update available voices for the new model
                    demo_instance.setup_voice_presets()
                    status_msg = f"✅ Successfully switched to model: {selected_model}"
                    print(status_msg)
                    new_voice_choices = list(demo_instance.available_voices.keys())
                    return (status_msg, render_model_select(model_choices, demo_instance.model_path)) + tuple(
                        render_speaker_select(
                            index,
                            new_voice_choices,
                            selected_speakers[index] if index < len(selected_speakers) else None,
                        )
                        for index in range(4)
                    )

                error_msg = f"❌ Failed to switch to model: {selected_model}"
                return (error_msg, render_model_select(model_choices, demo_instance.model_path)) + tuple(
                    gr.update() for _ in range(4)
                )
            except Exception as e:
                error_msg = f"❌ Error switching model: {str(e)}"
                print(error_msg)
                return (error_msg, render_model_select(model_choices, demo_instance.model_path)) + tuple(
                    gr.update() for _ in range(4)
                )

        # Connect model switching button
        load_model_btn.click(
            fn=switch_model,
            inputs=[model_selector] + speaker_selections,
            outputs=[log_output, model_selector] + speaker_selections,
            js=(
                f"(...values) => {{ {native_select_reader_js([model_select_id] + speaker_select_ids, [0, 1, 2, 3, 4])} "
                "return selected; }"
            ),
            queue=True, concurrency_id="model_operations", concurrency_limit=1
        )

        # Gain Control Event Handlers
        # Cache audio when complete audio changes (detects trimming)
        complete_audio_output.change(
            fn=cache_original_audio,
            inputs=[complete_audio_output],
            outputs=[]
        )
        
        # Apply gain when gain slider changes (uses cached original audio)
        gain_control.change(
            fn=lambda gain_db: apply_gain_to_complete_audio(None, gain_db),
            inputs=[gain_control],
            outputs=[complete_audio_output, log_output]
        )
        
        # Reset gain button
        gain_reset_btn.click(
            fn=reset_gain_control,
            outputs=[gain_control]
        )


    return interface


def convert_to_16_bit_wav(data):
    # Check if data is a tensor and move to cpu
    if torch.is_tensor(data):
        data = data.detach().cpu().numpy()
    
    # Ensure data is numpy array
    data = np.array(data)

    # Normalize to range [-1, 1] if it's not already
    if np.max(np.abs(data)) > 1.0:
        data = data / np.max(np.abs(data))
    
    # Scale to 16-bit integer range
    data = (data * 32767).astype(np.int16)
    return data


def parse_args():
    parser = argparse.ArgumentParser(description="VibeVoice Gradio Demo")
    parser.add_argument(
        "--model-path", "--model_path",
        dest="model_path",
        type=str,
        default=None,
        help="Model name or explicit local model directory",
    )
    add_model_cli_arguments(parser)
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device for inference",
    )
    parser.add_argument(
        "--inference_steps",
        type=int,
        default=10,
        help="Number of inference steps for DDPM (not exposed to users)",
    )
    parser.add_argument(
        "--share",
        action="store_true",
        help="Share the demo publicly via Gradio",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=7590,
        help="Port to run the demo on (always 7590 for network access)",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug logs for audio and vocal processing",
    )
    parser.add_argument(
        "--lod",
        action="store_true",
        help="Load On Demand: Load models in a disposable worker and release memory after each generation",
    )
    return parser.parse_args()


def main():
    """Main function to run the demo."""
    args = parse_args()
    model_settings = settings_from_args(args)
    model_path = args.model_path or default_model_name(
        legacy_default="WestZhang/VibeVoice-Large-pt",
    )

    # ⚠️ SECURITY WARNING: Check for --share flag
    if args.share:
        print("\n" + "="*80)
        print("🚨🚨🚨 SECURITY WARNING 🚨🚨🚨")
        print("="*80)
        print("⚠️  You are using the --share flag which will make your interface")
        print("⚠️  publicly accessible on the internet WITHOUT ANY PROTECTION!")
        print("⚠️  This is HIGHLY ADVISABLE NOT TO DO for security reasons.")
        print("⚠️  Anyone on the internet can access your model and generate audio.")
        print("⚠️  Consider using --port 7590 instead for local network access only.")
        print("="*80)
        print("🚨🚨🚨 PROCEEDING WITH PUBLIC SHARING ENABLED 🚨🚨🚨")
        print("="*80 + "\n")
        
        # Give user a chance to cancel
        try:
            response = input("Do you want to continue? (y/N): ").strip().lower()
            if response not in ['y', 'yes']:
                print("🛑 Sharing cancelled. Exiting...")
                return
        except KeyboardInterrupt:
            print("\n🛑 Sharing cancelled. Exiting...")
            return

    set_seed(42)  # Set a fixed seed for reproducibility

    print("🎙️ Initializing VibeVoice Demo with Streaming Support...")

    # Initialize demo instance
    demo_instance = VibeVoiceDemo(
        model_path=model_path,
        device=args.device,
        inference_steps=args.inference_steps,
        debug=args.debug,
        load_on_demand=args.lod,
        model_settings=model_settings,
    )
    
    # Create interface
    interface = create_demo_interface(demo_instance)
    
    print(f"🚀 Launching demo on port 7590 (network accessible)")
    print(f"📁 Model path: {model_path}")
    print(f"📂 Model directory: {model_settings.models_dir}")
    print(f"🎭 Available voices: {len(demo_instance.available_voices)}")
    print(f"🔴 Streaming mode: ENABLED")
    print(f"🔒 Session isolation: ENABLED")
    if args.debug:
        print(f"🔍 Debug mode: ENABLED (audio and vocal processing logs will be shown)")
    
    # Launch the interface
    try:
        queued = interface.queue(
            max_size=20,  # Maximum queue size
            default_concurrency_limit=1  # Process one request at a time
        )
        launch_compatibly(queued,
            share=args.share,
            server_port=7590,  # Always use port 7590 for network access
            server_name="0.0.0.0",  # Always serve on network interface
            show_error=True,
            show_api=False  # Hide API docs for cleaner interface
        )
    except KeyboardInterrupt:
        print("\n🛑 Shutting down gracefully...")
    except Exception as e:
        print(f"❌ Server error: {e}")
        raise


if __name__ == "__main__":
    # Set multiprocessing start method for CUDA compatibility
    # 'spawn' is required for CUDA to work correctly in child processes
    try:
        multiprocessing.set_start_method('spawn')
    except RuntimeError:
        # Already set, ignore
        pass
    
    main()
