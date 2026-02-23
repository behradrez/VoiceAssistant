"""Offline wake-word listener and command transcriber.

This script continuously listens for activation terms (for example, "listen bot").
After activation, it captures the next spoken instruction and stores it as text.
"""

from __future__ import annotations

import argparse
import json
import queue
import re
import sys
import time
from pathlib import Path
from typing import Optional

import sounddevice as sd
from vosk import KaldiRecognizer, Model


def normalize_text(text: str) -> str:
    text = text.lower().strip()
    return re.sub(r"\s+", " ", text)


class LocalVoiceAssistant:
    def __init__(
        self,
        model_path: str,
        activation_terms: list[str] | None = None,
        sample_rate: int = 16000,
        device: Optional[int] = None,
        silence_timeout: float = 1.3,
        max_command_seconds: float = 12.0,
    ) -> None:
        self.model = Model(model_path)
        self.sample_rate = sample_rate
        self.device = device
        self.silence_timeout = silence_timeout
        self.max_command_seconds = max_command_seconds
        self.activation_terms = [
            normalize_text(term) for term in (activation_terms or ["hey listen"])
        ]

        self.audio_queue: queue.Queue[bytes] = queue.Queue()
        self.latest_instruction: Optional[str] = None
        self.captured_instructions: list[str] = []

    def _new_recognizer(self) -> KaldiRecognizer:
        recognizer = KaldiRecognizer(self.model, self.sample_rate)
        recognizer.SetWords(False)
        return recognizer

    def _audio_callback(self, indata, frames, time_info, status) -> None:
        if status:
            print(f"[audio] {status}", file=sys.stderr)
        self.audio_queue.put(bytes(indata))

    def _detect_activation(self, text: str) -> tuple[Optional[str], str]:
        normalized = normalize_text(text)
        for term in self.activation_terms:
            idx = normalized.find(term)
            if idx != -1:
                tail = normalize_text(normalized[idx + len(term) :])
                return term, tail
        return None, ""

    def _finalize_instruction(
        self, command_rec: KaldiRecognizer, chunks: list[str]
    ) -> str:
        final_payload = json.loads(command_rec.FinalResult())
        final_text = normalize_text(final_payload.get("text", ""))
        if final_text:
            chunks.append(final_text)
        joined = normalize_text(" ".join(part for part in chunks if part))
        return joined

    def run_forever(self) -> None:
        wake_rec = self._new_recognizer()
        command_rec: Optional[KaldiRecognizer] = None
        command_chunks: list[str] = []
        activated = False
        command_started_at = 0.0
        last_speech_at: Optional[float] = None

        print(
            f"Listening for activation terms: {', '.join(self.activation_terms)}",
            flush=True,
        )

        with sd.RawInputStream(
            samplerate=self.sample_rate,
            blocksize=8000,
            dtype="int16",
            channels=1,
            device=self.device,
            callback=self._audio_callback,
        ):
            while True:
                data = self.audio_queue.get()
                now = time.monotonic()

                if not activated:
                    if wake_rec.AcceptWaveform(data):
                        payload = json.loads(wake_rec.Result())
                        observed_text = payload.get("text", "")
                        if observed_text:
                            print(f"[Observed] {observed_text}", flush=True)
                    else:
                        payload = json.loads(wake_rec.PartialResult())
                        observed_text = payload.get("partial", "")
                        if observed_text:
                            print(f"[Partial Observed] {observed_text}", flush=True)
                        
                    term, tail = self._detect_activation(observed_text)
                    if term:
                        activated = True
                        command_rec = self._new_recognizer()
                        command_chunks = [tail] if tail else []
                        command_started_at = now
                        last_speech_at = now if tail else None
                        print(f"Activated by '{term}'. Listening for instruction...")
                    continue

                assert command_rec is not None

                if command_rec.AcceptWaveform(data):
                    payload = json.loads(command_rec.Result())
                    text = normalize_text(payload.get("text", ""))
                    if text:
                        command_chunks.append(text)
                        last_speech_at = now
                else:
                    payload = json.loads(command_rec.PartialResult())
                    partial = normalize_text(payload.get("partial", ""))
                    if partial:
                        last_speech_at = now

                silence_elapsed = (
                    (now - last_speech_at) >= self.silence_timeout
                    if last_speech_at is not None
                    else False
                )
                command_timed_out = (now - command_started_at) >= self.max_command_seconds

                if silence_elapsed or command_timed_out:
                    instruction = self._finalize_instruction(command_rec, command_chunks)
                    if instruction:
                        self.latest_instruction = instruction
                        self.captured_instructions.append(instruction)
                        print(f"Instruction: {instruction}")
                    else:
                        print("No instruction detected. Returning to wake-word mode.")

                    activated = False
                    command_rec = None
                    command_chunks = []
                    command_started_at = 0.0
                    last_speech_at = None
                    wake_rec = self._new_recognizer()
                    print(
                        f"Listening for activation terms: {', '.join(self.activation_terms)}",
                        flush=True,
                    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Offline voice assistant wake-word + transcription loop"
    )
    parser.add_argument(
        "--model-path",
        default="model/vosk-model-small-en-us-0.15",
        help="Path to a local Vosk model directory",
    )
    parser.add_argument(
        "--activation-term",
        action="append",
        dest="activation_terms",
        help="Activation term to listen for. Can be passed multiple times.",
    )
    parser.add_argument(
        "--sample-rate",
        type=int,
        default=16000,
        help="Microphone sample rate",
    )
    parser.add_argument(
        "--device",
        type=int,
        default=None,
        help="Optional sounddevice input device index",
    )
    parser.add_argument(
        "--silence-timeout",
        type=float,
        default=1.3,
        help="Seconds of silence that finalize an instruction",
    )
    parser.add_argument(
        "--max-command-seconds",
        type=float,
        default=12.0,
        help="Hard timeout for one instruction after activation",
    )
    parser.add_argument(
        "--list-devices",
        action="store_true",
        help="Print available audio devices and exit",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.list_devices:
        print(sd.query_devices())
        return

    model_path = Path(args.model_path)
    if not model_path.exists():
        raise FileNotFoundError(f"Model path does not exist: {model_path}")

    assistant = LocalVoiceAssistant(
        model_path=str(model_path),
        activation_terms=args.activation_terms,
        sample_rate=args.sample_rate,
        device=args.device,
        silence_timeout=args.silence_timeout,
        max_command_seconds=args.max_command_seconds,
    )

    try:
        assistant.run_forever()
    except KeyboardInterrupt:
        print("\nStopping voice assistant.")
        if assistant.latest_instruction:
            print(f"Last captured instruction: {assistant.latest_instruction}")


if __name__ == "__main__":
    main()
