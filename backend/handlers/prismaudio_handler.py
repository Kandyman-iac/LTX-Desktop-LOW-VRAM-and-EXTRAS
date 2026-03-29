"""PrismAudio handler — runs ThinkSound/PrismAudio video-to-audio inference natively on Windows."""

from __future__ import annotations

import glob
import logging
import shutil
import subprocess
import tempfile
import threading
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from subprocess import PIPE, STDOUT
from typing import Literal

from _routes._errors import HTTPError
from api_types import PrismAudioGenerateRequest, PrismAudioProgressResponse

logger = logging.getLogger(__name__)

PrismAudioStatus = Literal["idle", "running", "complete", "error", "cancelled"]

# ---------------------------------------------------------------------------
# Configurable constants — adjust to match your local PrismAudio install.
# ---------------------------------------------------------------------------
_PRISMAUDIO_DIR = "C:/AI/ThinkSound"
_PRISMAUDIO_CONDA_ENV = "prismaudio"
_PRISMAUDIO_PYTHON: str | None = None  # e.g. r"C:\miniconda3\envs\prismaudio\python.exe"
_MAX_LOG_LINES = 200


@dataclass
class _PrismAudioJob:
    status: PrismAudioStatus = "idle"
    output_path: str | None = None
    error: str | None = None
    log_lines: list[str] = field(default_factory=list)
    process: subprocess.Popen | None = None  # type: ignore[type-arg]


class PrismAudioHandler:
    """Runs PrismAudio (ThinkSound) video-to-audio inference in a background thread on Windows."""

    def __init__(self, outputs_dir: Path) -> None:
        self._outputs_dir = outputs_dir
        self._lock = threading.Lock()
        self._job = _PrismAudioJob()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def get_progress(self) -> PrismAudioProgressResponse:
        with self._lock:
            j = self._job
            return PrismAudioProgressResponse(
                status=j.status,
                output_path=j.output_path,
                error=j.error,
                log_tail="\n".join(j.log_lines[-50:]),
            )

    def generate(self, req: PrismAudioGenerateRequest) -> PrismAudioProgressResponse:
        with self._lock:
            if self._job.status == "running":
                raise HTTPError(409, "PrismAudio generation already in progress")
            self._job = _PrismAudioJob(status="running")

        thread = threading.Thread(target=self._run, args=(req,), daemon=True)
        thread.start()
        return self.get_progress()

    def cancel(self) -> None:
        with self._lock:
            if self._job.process and self._job.status == "running":
                try:
                    self._job.process.terminate()
                except Exception:
                    pass
                self._job.status = "cancelled"

    # ------------------------------------------------------------------
    # Background worker
    # ------------------------------------------------------------------

    def _run(self, req: PrismAudioGenerateRequest) -> None:
        stem = f"prismaudio_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:6]}"
        out_dir = tempfile.mkdtemp(prefix="prismaudio_")

        # Build the inference command.
        if _PRISMAUDIO_PYTHON:
            inner_cmd: list[str] = [
                _PRISMAUDIO_PYTHON,
                "infer.py",
                "--video_path", req.video_path,
                "--save_dir", out_dir,
            ]
        else:
            inner_cmd = [
                "conda", "run",
                "-n", _PRISMAUDIO_CONDA_ENV,
                "--no-capture-output",
                "python", "infer.py",
                "--video_path", req.video_path,
                "--save_dir", out_dir,
            ]

        if req.prompt.strip():
            inner_cmd += ["--text_prompt", req.prompt]
        if req.seed is not None:
            inner_cmd += ["--seed", str(req.seed)]

        logger.info(
            "[prismaudio] Starting: video=%s prompt=%r seed=%s",
            req.video_path, req.prompt, req.seed,
        )

        try:
            process = subprocess.Popen(
                inner_cmd,
                cwd=_PRISMAUDIO_DIR,
                stdout=PIPE,
                stderr=STDOUT,
                text=True,
                bufsize=1,
                encoding="utf-8",
                errors="replace",
            )
            with self._lock:
                self._job.process = process

            assert process.stdout is not None
            for line in process.stdout:
                line = line.rstrip()
                logger.info("[prismaudio] %s", line)
                with self._lock:
                    self._job.log_lines.append(line)
                    if len(self._job.log_lines) > _MAX_LOG_LINES:
                        self._job.log_lines = self._job.log_lines[-_MAX_LOG_LINES:]

            process.wait()

            with self._lock:
                if self._job.status == "cancelled":
                    shutil.rmtree(out_dir, ignore_errors=True)
                    return
                if process.returncode != 0:
                    self._job.status = "error"
                    self._job.error = f"PrismAudio exited with code {process.returncode}"
                    shutil.rmtree(out_dir, ignore_errors=True)
                    return

            # Find output MP4 (PrismAudio writes a merged WAV + MP4 to save_dir).
            matches = glob.glob(f"{out_dir}/*.mp4")
            if not matches:
                shutil.rmtree(out_dir, ignore_errors=True)
                with self._lock:
                    self._job.status = "error"
                    self._job.error = "PrismAudio output MP4 not found in temp directory"
                return

            win_out = self._outputs_dir / f"{stem}.mp4"
            shutil.copy2(matches[0], win_out)

            # Clean up temp dir now that we have the file.
            shutil.rmtree(out_dir, ignore_errors=True)

            with self._lock:
                self._job.status = "complete"
                self._job.output_path = str(win_out)
                self._job.log_lines.append(f"✓ Output: {win_out}")

        except Exception as exc:
            logger.exception("[prismaudio] Unexpected error")
            shutil.rmtree(out_dir, ignore_errors=True)
            with self._lock:
                self._job.status = "error"
                self._job.error = str(exc)
