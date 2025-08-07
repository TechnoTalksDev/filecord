from pydantic import BaseModel
from enum import Enum
from datetime import datetime
from typing import Optional

class JobStatus(Enum):
    QUEUED = "queued"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"

class ConversionJob(BaseModel):
    id: str
    filename: str
    upload_path: str
    output_path: Optional[str] = None
    status: JobStatus = JobStatus.QUEUED
    progress: int = 0
    message: str = ""
    error: Optional[str] = None
    created_at: datetime
    queue_position: Optional[int] = None
    queue_size: Optional[int] = None

class ProgressUpdate(BaseModel):
    progress: int
    message: str
    output_path: Optional[str] = None

class JobResponse(BaseModel):
    job_id: str
    message: str

class StatusResponse(BaseModel):
    id: str
    filename: str
    status: str
    progress: int
    message: str
    error: Optional[str] = None
    created_at: str
    queue_position: Optional[int] = None
    queue_size: Optional[int] = None
