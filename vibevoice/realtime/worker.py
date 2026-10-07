"""Disposable realtime worker with bounded chunk IPC."""

import multiprocessing
import queue
import threading
import time

from vibevoice.realtime.service import RealtimeService


def _send(output, cancel, kind, value=None):
    while not cancel.is_set():
        try:
            output.put((kind, value), timeout=0.1)
            return
        except queue.Full:
            continue


def _realtime_worker(settings, device, selection, text, parameters, output, cancel):
    try:
        service = RealtimeService(settings, device)
        service.load(*selection)
        if cancel.is_set():
            return
        _send(output, cancel, "loaded")
        for chunk in service.stream(text, **parameters, cancel_event=cancel):
            _send(output, cancel, "chunk", chunk)
        _send(output, cancel, "done")
    except Exception as exc:  # noqa: BLE001 - report worker failures over IPC
        _send(output, cancel, "error", str(exc))


class RealtimeWorkerService:
    def __init__(self, settings, device):
        self.settings, self.device = settings, device
        self.selection = None
        self.context = multiprocessing.get_context("spawn")
        self.stop_event = self.context.Event()
        self.lock = threading.Lock()
        self.process = None

    def load(self, model_path, voice_path):
        if self.lock.locked():
            raise RuntimeError("Stop realtime generation before changing the model")
        self.selection = (str(model_path), str(voice_path))

    def stop(self):
        self.stop_event.set()

    def unload(self):
        if self.lock.locked():
            raise RuntimeError("Stop realtime generation before unloading the worker")
        self.selection = None

    def stream(self, text, *, cancel_event=None, **parameters):
        if self.selection is None:
            raise ValueError("Select a realtime model and voice preset first")
        if not self.lock.acquire(blocking=False):
            raise RuntimeError("Realtime generation is already active")
        output = self.context.Queue(maxsize=4)
        self.stop_event.clear()
        try:
            if cancel_event is not None and cancel_event.is_set():
                return
            self.process = self.context.Process(
                target=_realtime_worker,
                args=(
                    self.settings,
                    self.device,
                    self.selection,
                    text,
                    parameters,
                    output,
                    self.stop_event,
                ),
            )
            self.process.start()
            last_message = time.monotonic()
            loaded = False
            while not self.stop_event.is_set():
                if cancel_event is not None and cancel_event.is_set():
                    return
                try:
                    kind, data = output.get(timeout=0.1)
                except queue.Empty:
                    if not self.process.is_alive():
                        raise RuntimeError(
                            f"Realtime worker exited without completion (exit {self.process.exitcode})"
                        )
                    if loaded and time.monotonic() - last_message > 180:
                        raise RuntimeError(
                            "Realtime worker timed out without producing audio"
                        )
                    continue
                last_message = time.monotonic()
                if kind == "loaded":
                    loaded = True
                elif kind == "chunk":
                    yield data
                elif kind == "done":
                    return
                elif kind == "error":
                    raise RuntimeError(data)
                else:
                    raise RuntimeError(f"Unknown realtime worker message: {kind}")
        finally:
            self.stop_event.set()
            if self.process is not None:
                if self.process.pid is not None:
                    self.process.join(timeout=2)
                    if self.process.is_alive():
                        self.process.terminate()
                        self.process.join(timeout=2)
                    if self.process.is_alive():
                        self.process.kill()
                        self.process.join()
                self.process.close()
                self.process = None
            output.cancel_join_thread()
            output.close()
            self.lock.release()
