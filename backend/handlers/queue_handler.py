"""Generation queue — processes video generation jobs sequentially."""

from __future__ import annotations

import logging
import threading
import time
import uuid
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Literal

from api_types import GenerateVideoRequest

if TYPE_CHECKING:
    from handlers.video_generation_handler import VideoGenerationHandler

logger = logging.getLogger(__name__)


@dataclass
class QueuedJob:
    id: str
    request: GenerateVideoRequest
    status: Literal["pending", "running", "complete", "error", "cancelled"]
    result_path: str | None = None
    error: str | None = None
    created_at: float = field(default_factory=time.time)


class QueueHandler:
    """Thread-safe queue that processes video generation requests one at a time."""

    def __init__(self) -> None:
        self._jobs: list[QueuedJob] = []
        self._lock = threading.Lock()
        self._video_generation: VideoGenerationHandler | None = None

    def set_video_generation(self, handler: VideoGenerationHandler) -> None:
        self._video_generation = handler

    def add_job(self, req: GenerateVideoRequest) -> QueuedJob:
        job = QueuedJob(id=str(uuid.uuid4()), request=req, status="pending")
        with self._lock:
            self._jobs.append(job)
        logger.info("Queue: job %s added (queue length=%d)", job.id[:8], len(self._jobs))
        self._maybe_start_next()
        return job

    def get_jobs(self) -> list[QueuedJob]:
        with self._lock:
            return list(self._jobs)

    def remove_job(self, job_id: str) -> bool:
        with self._lock:
            for job in self._jobs:
                if job.id == job_id and job.status == "pending":
                    job.status = "cancelled"
                    logger.info("Queue: job %s cancelled", job_id[:8])
                    return True
        return False

    def _maybe_start_next(self) -> None:
        with self._lock:
            if any(j.status == "running" for j in self._jobs):
                return
            next_job = next((j for j in self._jobs if j.status == "pending"), None)
            if next_job is None:
                return
            next_job.status = "running"
        threading.Thread(target=self._process_job, args=(next_job,), daemon=True).start()

    def _process_job(self, job: QueuedJob) -> None:
        logger.info("Queue: processing job %s (prompt=%.40s...)", job.id[:8], job.request.prompt)
        try:
            if self._video_generation is None:
                raise RuntimeError("QueueHandler not wired to VideoGenerationHandler")
            result = self._video_generation.generate(job.request)
            with self._lock:
                job.status = "complete"
                job.result_path = result.video_path
            logger.info("Queue: job %s complete — %s", job.id[:8], result.video_path)
        except Exception as exc:
            with self._lock:
                job.status = "error"
                job.error = str(exc)
            logger.warning("Queue: job %s failed — %s", job.id[:8], exc)
        finally:
            self._maybe_start_next()
