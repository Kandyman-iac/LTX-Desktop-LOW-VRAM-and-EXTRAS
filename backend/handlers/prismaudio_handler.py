"""PrismAudio handler — runs ThinkSound/PrismAudio video-to-audio inference.

Supports two modes controlled by _USE_WSL:
  False (default) — native Windows, uses conda env or direct Python executable
  True            — WSL2 subprocess (same pattern as MMAudio/MagiHuman)
"""

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

# Set True to run PrismAudio inside WSL2 (Linux) instead of native Windows.
_USE_WSL = False

# --- Windows-native settings (used when _USE_WSL = False) ---
_PRISMAUDIO_DIR_WIN = "C:/AI/ThinkSound"
_PRISMAUDIO_CONDA_ENV = "prismaudio"
# Set to a direct Python path to skip conda run entirely, e.g.:
#   r"C:\miniconda3\envs\prismaudio\python.exe"
_PRISMAUDIO_PYTHON_WIN: str | None = None

# --- WSL settings (used when _USE_WSL = True) ---
_WSL_DISTRO = "Ubuntu-24.04"
_WSL_USER = "mike_hunt"
_PRISMAUDIO_DIR_WSL = "/home/mike_hunt/ThinkSound"
_PRISMAUDIO_CONDA_ENV_WSL = "prismaudio"
# Set to a direct Python path inside WSL to skip conda run, e.g.:
#   "/home/mike_hunt/miniconda3/envs/prismaudio/bin/python"
_PRISMAUDIO_PYTHON_WSL: str | None = None

_MAX_LOG_LINES = 200


@dataclass
class _PrismAudioJob:
    status: PrismAudioStatus = "idle"
    output_path: str | None = None
    error: str | None = None
    log_lines: list[str] = field(default_factory=list)
    process: subprocess.Popen | None = None  # type: ignore[type-arg]


class PrismAudioHandler:
    """Runs PrismAudio (ThinkSound) video-to-audio inference in a background thread."""

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
        if _USE_WSL:
            self._run_wsl(req)
        else:
            self._run_windows(req)

    # ------------------------------------------------------------------
    # Windows-native path
    # ------------------------------------------------------------------

    def _run_windows(self, req: PrismAudioGenerateRequest) -> None:
        stem = f"prismaudio_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:6]}"
        out_dir = tempfile.mkdtemp(prefix="prismaudio_")

        if _PRISMAUDIO_PYTHON_WIN:
            cmd: list[str] = [_PRISMAUDIO_PYTHON_WIN, "infer.py"]
        else:
            cmd = ["conda", "run", "-n", _PRISMAUDIO_CONDA_ENV, "--no-capture-output", "python", "infer.py"]

        cmd += ["--video_path", req.video_path, "--save_dir", out_dir]
        if req.prompt.strip():
            cmd += ["--text_prompt", req.prompt]
        if req.seed is not None:
            cmd += ["--seed", str(req.seed)]

        logger.info("[prismaudio/win] Starting: video=%s prompt=%r", req.video_path, req.prompt)

        try:
            process = subprocess.Popen(
                cmd,
                cwd=_PRISMAUDIO_DIR_WIN,
                stdout=PIPE, stderr=STDOUT,
                text=True, bufsize=1, encoding="utf-8", errors="replace",
            )
            with self._lock:
                self._job.process = process

            assert process.stdout is not None
            for line in process.stdout:
                self._append_log(line.rstrip())

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

            self._copy_output(out_dir, stem)

        except Exception as exc:
            logger.exception("[prismaudio/win] Unexpected error")
            shutil.rmtree(out_dir, ignore_errors=True)
            with self._lock:
                self._job.status = "error"
                self._job.error = str(exc)

    # ------------------------------------------------------------------
    # WSL path
    # ------------------------------------------------------------------

    def _run_wsl(self, req: PrismAudioGenerateRequest) -> None:
        stem = f"prismaudio_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:6]}"
        wsl_out_dir = f"/tmp/prismaudio_out_{stem}"
        wsl_video_path = _win_to_wsl_path(req.video_path)

        if _PRISMAUDIO_PYTHON_WSL:
            py_prefix = _PRISMAUDIO_PYTHON_WSL
        else:
            py_prefix = f"conda run -n {_PRISMAUDIO_CONDA_ENV_WSL} --no-capture-output python"

        # Escape single quotes in prompt
        safe_prompt = req.prompt.replace("'", "'\\''")
        inner = (
            f"mkdir -p '{wsl_out_dir}' && "
            f"cd '{_PRISMAUDIO_DIR_WSL}' && "
            f"{py_prefix} infer.py"
            f" --video_path '{wsl_video_path}'"
            f" --save_dir '{wsl_out_dir}'"
        )
        if req.prompt.strip():
            inner += f" --text_prompt '{safe_prompt}'"
        if req.seed is not None:
            inner += f" --seed {req.seed}"

        cmd = ["wsl", "-d", _WSL_DISTRO, "-u", _WSL_USER, "bash", "-c", inner]
        logger.info("[prismaudio/wsl] Starting: video=%s prompt=%r", req.video_path, req.prompt)

        try:
            process = subprocess.Popen(
                cmd,
                stdout=PIPE, stderr=STDOUT,
                text=True, bufsize=1, encoding="utf-8", errors="replace",
            )
            with self._lock:
                self._job.process = process

            assert process.stdout is not None
            for line in process.stdout:
                self._append_log(line.rstrip())

            process.wait()

            with self._lock:
                if self._job.status == "cancelled":
                    return
                if process.returncode != 0:
                    self._job.status = "error"
                    self._job.error = f"PrismAudio exited with code {process.returncode}"
                    return

            # Copy output from WSL to Windows outputs dir
            win_out = self._outputs_dir / f"{stem}.mp4"
            win_wsl = _win_to_wsl_path(str(win_out))

            copy_bash = (
                f"f=$(ls -t '{wsl_out_dir}'/*.mp4 2>/dev/null | head -1); "
                f"[ -n \"$f\" ] && cp \"$f\" '{win_wsl}' && echo OK || echo NOTFOUND"
            )
            cp = subprocess.run(
                ["wsl", "-d", _WSL_DISTRO, "-u", _WSL_USER, "bash", "-c", copy_bash],
                capture_output=True, text=True,
            )
            # Cleanup WSL temp dir
            subprocess.run(
                ["wsl", "-d", _WSL_DISTRO, "-u", _WSL_USER, "bash", "-c", f"rm -rf '{wsl_out_dir}'"],
                capture_output=True,
            )
            if "OK" not in cp.stdout:
                with self._lock:
                    self._job.status = "error"
                    self._job.error = f"PrismAudio output not found after generation. stderr: {cp.stderr.strip()}"
                return

            with self._lock:
                self._job.status = "complete"
                self._job.output_path = str(win_out)
                self._job.log_lines.append(f"✓ Output: {win_out}")

        except Exception as exc:
            logger.exception("[prismaudio/wsl] Unexpected error")
            with self._lock:
                self._job.status = "error"
                self._job.error = str(exc)

    # ------------------------------------------------------------------
    # Shared helpers
    # ------------------------------------------------------------------

    def _append_log(self, line: str) -> None:
        logger.info("[prismaudio] %s", line)
        with self._lock:
            self._job.log_lines.append(line)
            if len(self._job.log_lines) > _MAX_LOG_LINES:
                self._job.log_lines = self._job.log_lines[-_MAX_LOG_LINES:]

    def _copy_output(self, out_dir: str, stem: str) -> None:
        """Find the output MP4, copy to outputs_dir, clean up temp dir."""
        matches = glob.glob(f"{out_dir}/*.mp4")
        if not matches:
            shutil.rmtree(out_dir, ignore_errors=True)
            with self._lock:
                self._job.status = "error"
                self._job.error = "PrismAudio output MP4 not found in temp directory"
            return

        win_out = self._outputs_dir / f"{stem}.mp4"
        shutil.copy2(matches[0], win_out)
        shutil.rmtree(out_dir, ignore_errors=True)

        with self._lock:
            self._job.status = "complete"
            self._job.output_path = str(win_out)
            self._job.log_lines.append(f"✓ Output: {win_out}")


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------

def _win_to_wsl_path(win_path: str) -> str:
    """Convert C:\\Users\\... → /mnt/c/Users/..."""
    p = win_path.replace("\\", "/")
    if len(p) >= 2 and p[1] == ":":
        drive = p[0].lower()
        p = f"/mnt/{drive}{p[2:]}"
    return p
