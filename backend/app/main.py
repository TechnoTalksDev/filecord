from fastapi import FastAPI, File, UploadFile, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, FileResponse
from fastapi.staticfiles import StaticFiles
import uvicorn
import asyncio
import uuid
import os
import shutil
from pathlib import Path
from typing import Dict, Optional, List
import json
from datetime import datetime
import mimetypes
import logging
import traceback

from .services.video_processor import VideoProcessor
from .services.file_manager import FileManager
from .models import ConversionJob, JobStatus

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class QueueManager:
    """Manages the video processing queue to ensure one job at a time"""
    
    def __init__(self):
        self.queue = asyncio.Queue()
        self.job_order: List[str] = []  # Track job order for position calculation
        self.current_job_id: Optional[str] = None
        self.is_processing = False
        self._lock = asyncio.Lock()
    
    async def add_job(self, job_id: str):
        """Add a job to the queue"""
        async with self._lock:
            await self.queue.put(job_id)
            self.job_order.append(job_id)
            logger.info(f"Added job {job_id} to queue. Queue size: {len(self.job_order)}")
    
    def get_queue_position(self, job_id: str) -> tuple[Optional[int], int]:
        """Get queue position and total queue size for a job"""
        if job_id == self.current_job_id:
            return None, len(self.job_order)  # Currently processing
        
        try:
            position = self.job_order.index(job_id) + 1
            if self.current_job_id is not None:
                position += 1  # Add 1 if there's a job currently processing
            return position, len(self.job_order) + (1 if self.current_job_id else 0)
        except ValueError:
            return None, len(self.job_order)  # Job not in queue
    
    async def remove_job_from_order(self, job_id: str):
        """Remove job from order tracking"""
        async with self._lock:
            if job_id in self.job_order:
                self.job_order.remove(job_id)
                logger.info(f"Removed job {job_id} from queue order. Queue size: {len(self.job_order)}")
    
    async def process_queue(self):
        """Main queue processor - runs continuously"""
        logger.info("Queue processor started")
        
        while True:
            try:
                # Wait for a job in the queue
                job_id = await self.queue.get()
                
                # Update current job tracking
                self.current_job_id = job_id
                self.is_processing = True
                await self.remove_job_from_order(job_id)
                
                logger.info(f"Starting to process job {job_id}")
                
                # Process the job
                await process_video_background(job_id)
                
                # Mark task as done
                self.queue.task_done()
                
            except Exception as e:
                logger.error(f"Error in queue processor: {str(e)}")
                logger.error(traceback.format_exc())
                self.queue.task_done()
            finally:
                # Reset current job tracking
                self.current_job_id = None
                self.is_processing = False
                logger.info("Finished processing job, ready for next")


# Create global queue manager
queue_manager = QueueManager()

# Create FastAPI app
app = FastAPI(
    title="FileCord Video Converter",
    description="Convert videos to 10MB or less",
    version="1.0.0"
)

# Configure CORS for frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify your frontend URL
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize services
video_processor = VideoProcessor()
file_manager = FileManager()

# In-memory job storage (use Redis in production)
jobs: Dict[str, ConversionJob] = {}

# Ensure directories exist
os.makedirs("uploads", exist_ok=True)
os.makedirs("outputs", exist_ok=True)

# Global variable to store the queue task
queue_task: Optional[asyncio.Task] = None

def cleanup_job_files(job_id: str) -> dict:
    """
    Clean up files for a job. Returns a dict with cleanup results.
    This function is used both by the cleanup endpoint and background tasks.
    """
    logger.info(f"Starting cleanup for job {job_id}")
    
    if job_id not in jobs:
        logger.warning(f"Cleanup requested for non-existent job {job_id}")
        return {
            "success": False,
            "message": "Job not found",
            "files_cleaned": [],
            "cleanup_errors": ["Job not found"]
        }
    
    job = jobs[job_id]
    cleanup_errors = []
    files_cleaned = []
    
    # Clean up upload file
    if job.upload_path:
        logger.info(f"Attempting to clean up upload file: {job.upload_path}")
        if os.path.exists(job.upload_path):
            try:
                logger.info(f"Upload file exists, attempting deletion: {job.upload_path}")
                os.remove(job.upload_path)
                
                # Verify file was actually deleted
                if os.path.exists(job.upload_path):
                    error_msg = f"Upload file still exists after deletion attempt: {job.upload_path}"
                    logger.error(error_msg)
                    cleanup_errors.append(error_msg)
                else:
                    logger.info(f"Successfully deleted upload file: {job.upload_path}")
                    files_cleaned.append(job.upload_path)
                    
            except OSError as e:
                error_msg = f"Failed to delete upload file {job.upload_path}: {str(e)}"
                logger.error(error_msg)
                cleanup_errors.append(error_msg)
            except Exception as e:
                error_msg = f"Unexpected error deleting upload file {job.upload_path}: {str(e)}"
                logger.error(error_msg)
                cleanup_errors.append(error_msg)
        else:
            logger.info(f"Upload file does not exist (already cleaned?): {job.upload_path}")
    else:
        logger.info("No upload path specified for cleanup")
    
    # Clean up output file
    if job.output_path:
        logger.info(f"Attempting to clean up output file: {job.output_path}")
        if os.path.exists(job.output_path):
            try:
                logger.info(f"Output file exists, attempting deletion: {job.output_path}")
                os.remove(job.output_path)
                
                # Verify file was actually deleted
                if os.path.exists(job.output_path):
                    error_msg = f"Output file still exists after deletion attempt: {job.output_path}"
                    logger.error(error_msg)
                    cleanup_errors.append(error_msg)
                else:
                    logger.info(f"Successfully deleted output file: {job.output_path}")
                    files_cleaned.append(job.output_path)
                    
            except OSError as e:
                error_msg = f"Failed to delete output file {job.output_path}: {str(e)}"
                logger.error(error_msg)
                cleanup_errors.append(error_msg)
            except Exception as e:
                error_msg = f"Unexpected error deleting output file {job.output_path}: {str(e)}"
                logger.error(error_msg)
                cleanup_errors.append(error_msg)
        else:
            logger.info(f"Output file does not exist (already cleaned?): {job.output_path}")
    else:
        logger.info("No output path specified for cleanup")
    
    # Remove job from memory
    try:
        del jobs[job_id]
        logger.info(f"Removed job {job_id} from memory")
    except KeyError:
        logger.warning(f"Job {job_id} was already removed from memory")
    
    # Prepare response
    success = len(cleanup_errors) == 0
    if cleanup_errors:
        logger.warning(f"Cleanup completed with errors for job {job_id}: {cleanup_errors}")
        message = f"Cleanup completed with {len(cleanup_errors)} error(s)"
    else:
        logger.info(f"Cleanup completed successfully for job {job_id}")
        message = "Files cleaned up successfully"
    
    return {
        "success": success,
        "message": message,
        "files_cleaned": files_cleaned,
        "cleanup_errors": cleanup_errors
    }

def background_cleanup_job(job_id: str):
    """
    Background task wrapper for cleanup that handles errors gracefully.
    This ensures background task errors don't crash the application.
    """
    try:
        result = cleanup_job_files(job_id)
        if result["success"]:
            logger.info(f"Background cleanup completed successfully for job {job_id}")
        else:
            logger.warning(f"Background cleanup completed with errors for job {job_id}: {result['cleanup_errors']}")
    except Exception as e:
        logger.error(f"Background cleanup failed for job {job_id}: {str(e)}")
        logger.error(traceback.format_exc())

@app.on_event("startup")
async def startup_event():
    """Start the queue processor when the app starts"""
    global queue_task
    queue_task = asyncio.create_task(queue_manager.process_queue())
    logger.info("Queue processor task started")

@app.on_event("shutdown")
async def shutdown_event():
    """Clean shutdown of the queue processor"""
    global queue_task
    if queue_task:
        queue_task.cancel()
        try:
            await queue_task
        except asyncio.CancelledError:
            logger.info("Queue processor task cancelled")
    logger.info("App shutdown complete")

@app.get("/")
async def root():
    return {"message": "FileCord Video Converter API", "status": "running"}

@app.post("/upload")
async def upload_video(
    file: UploadFile = File(...)
):
    """Upload a video file and add to conversion queue"""
    
    logger.info(f"Received upload request for file: {file.filename}, content_type: {file.content_type}")
    
    # Validate file type
    if not file.content_type or not file.content_type.startswith('video/'):
        logger.warning(f"Invalid file type: {file.content_type}")
        raise HTTPException(status_code=400, detail="File must be a video")
    
    # Generate job ID
    job_id = str(uuid.uuid4())
    logger.info(f"Generated job ID: {job_id}")
    
    # Save uploaded file
    upload_path = f"uploads/{job_id}_{file.filename}"
    try:
        with open(upload_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        logger.info(f"File saved to: {upload_path}")
    except Exception as e:
        logger.error(f"Failed to save file: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to save file: {str(e)}")
    
    # Create job entry
    job = ConversionJob(
        id=job_id,
        filename=file.filename,
        upload_path=upload_path,
        status=JobStatus.QUEUED,
        created_at=datetime.now(),
        message="Added to conversion queue"
    )
    jobs[job_id] = job
    logger.info(f"Created job entry for {job_id}")
    
    # Add to queue instead of starting background task
    await queue_manager.add_job(job_id)
    logger.info(f"Added job {job_id} to processing queue")
    
    return {"job_id": job_id, "message": "Video uploaded successfully, added to conversion queue"}

@app.get("/progress/{job_id}")
async def get_progress(job_id: str):
    """Server-Sent Events endpoint for progress updates"""
    
    if job_id not in jobs:
        raise HTTPException(status_code=404, detail="Job not found")
    
    async def event_generator():
        while True:
            if job_id not in jobs:
                break
                
            job = jobs[job_id]
            
            # Get queue information
            queue_position, queue_size = queue_manager.get_queue_position(job_id)
            
            # Update job queue information
            job.queue_position = queue_position
            job.queue_size = queue_size
            
            # Send progress update
            data = {
                "status": job.status.value,
                "progress": job.progress,
                "message": job.message,
                "error": job.error,
                "queue_position": queue_position,
                "queue_size": queue_size
            }
            
            yield f"data: {json.dumps(data)}\n\n"
            
            # Break if job is complete or failed
            if job.status in [JobStatus.COMPLETED, JobStatus.FAILED]:
                break
                
            await asyncio.sleep(0.5)  # Update every 500ms
    
    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
        }
    )

@app.get("/download/{job_id}")
async def download_video(job_id: str, background_tasks: BackgroundTasks):
    """Download the converted video and automatically cleanup files afterward"""
    
    if job_id not in jobs:
        raise HTTPException(status_code=404, detail="Job not found")
    
    job = jobs[job_id]
    
    if job.status != JobStatus.COMPLETED:
        raise HTTPException(status_code=400, detail="Conversion not completed")
    
    if not job.output_path or not os.path.exists(job.output_path):
        raise HTTPException(status_code=404, detail="Output file not found")
    
    # Schedule automatic cleanup after download completes
    background_tasks.add_task(background_cleanup_job, job_id)
    logger.info(f"Scheduled automatic cleanup for job {job_id} after download")
    
    # Determine the filename for download
    filename = f"converted_{job.filename}"
    
    return FileResponse(
        job.output_path,
        filename=filename,
        media_type="video/mp4"
    )

@app.delete("/cleanup/{job_id}")
async def cleanup_files(job_id: str):
    """Manually clean up temporary files for a job"""
    
    result = cleanup_job_files(job_id)
    
    if not result["success"] and "Job not found" in result["cleanup_errors"]:
        raise HTTPException(status_code=404, detail="Job not found")
    
    # Return the cleanup result
    response = {
        "message": result["message"],
        "files_cleaned": result["files_cleaned"],
        "cleanup_errors": result["cleanup_errors"]
    }
    
    return response

@app.get("/status/{job_id}")
async def get_job_status(job_id: str):
    """Get job status"""
    
    if job_id not in jobs:
        raise HTTPException(status_code=404, detail="Job not found")
    
    job = jobs[job_id]
    
    # Get current queue information
    queue_position, queue_size = queue_manager.get_queue_position(job_id)
    
    return {
        "id": job.id,
        "filename": job.filename,
        "status": job.status.value,
        "progress": job.progress,
        "message": job.message,
        "error": job.error,
        "created_at": job.created_at.isoformat(),
        "queue_position": queue_position,
        "queue_size": queue_size
    }

async def process_video_background(job_id: str):
    """Background task to process video - now called by queue processor"""
    
    logger.info(f"Starting background processing for job {job_id}")
    
    if job_id not in jobs:
        logger.error(f"Job {job_id} not found in jobs dictionary")
        return
        
    job = jobs[job_id]
    
    try:
        job.status = JobStatus.PROCESSING
        job.message = "Starting video processing..."
        job.queue_position = None  # No longer in queue
        logger.info(f"Job {job_id} status set to PROCESSING")
        
        # Process the video with progress updates
        logger.info(f"Starting video conversion for {job_id}: {job.upload_path}")
        async for progress_update in video_processor.convert_video(
            job.upload_path, 
            job_id,
            target_size_mb=10
        ):
            logger.debug(f"Job {job_id} progress update: {progress_update}")
            job.progress = progress_update["progress"]
            job.message = progress_update["message"]
            
            if "output_path" in progress_update:
                job.output_path = progress_update["output_path"]
                logger.info(f"Job {job_id} output path set: {job.output_path}")
        
        job.status = JobStatus.COMPLETED
        job.progress = 100
        job.message = "Video conversion completed successfully!"
        logger.info(f"Job {job_id} completed successfully")
        
    except Exception as e:
        logger.error(f"Job {job_id} failed with error: {str(e)}")
        logger.error(f"Full traceback for job {job_id}:")
        logger.error(traceback.format_exc())
        
        job.status = JobStatus.FAILED
        job.error = str(e)
        job.message = f"Conversion failed: {str(e)}"
        
        # Also log the job state for debugging
        logger.error(f"Job {job_id} final state - Status: {job.status}, Error: {job.error}, Message: {job.message}")

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
