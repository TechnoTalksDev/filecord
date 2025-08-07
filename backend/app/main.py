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
from typing import Dict, Optional
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

@app.get("/")
async def root():
    return {"message": "FileCord Video Converter API", "status": "running"}

@app.post("/upload")
async def upload_video(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...)
):
    """Upload a video file and start conversion process"""
    
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
        created_at=datetime.now()
    )
    jobs[job_id] = job
    logger.info(f"Created job entry for {job_id}")
    
    # Start conversion in background
    background_tasks.add_task(process_video_background, job_id)
    logger.info(f"Started background task for {job_id}")
    
    return {"job_id": job_id, "message": "Video uploaded successfully, conversion started"}

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
            
            # Send progress update
            data = {
                "status": job.status.value,
                "progress": job.progress,
                "message": job.message,
                "error": job.error
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
async def download_video(job_id: str):
    """Download the converted video"""
    
    if job_id not in jobs:
        raise HTTPException(status_code=404, detail="Job not found")
    
    job = jobs[job_id]
    
    if job.status != JobStatus.COMPLETED:
        raise HTTPException(status_code=400, detail="Conversion not completed")
    
    if not job.output_path or not os.path.exists(job.output_path):
        raise HTTPException(status_code=404, detail="Output file not found")
    
    # Determine the filename for download
    filename = f"converted_{job.filename}"
    
    return FileResponse(
        job.output_path,
        filename=filename,
        media_type="video/mp4"
    )

@app.delete("/cleanup/{job_id}")
async def cleanup_files(job_id: str):
    """Clean up temporary files for a job"""
    
    if job_id not in jobs:
        raise HTTPException(status_code=404, detail="Job not found")
    
    job = jobs[job_id]
    
    # Clean up files
    if job.upload_path and os.path.exists(job.upload_path):
        os.remove(job.upload_path)
    
    if job.output_path and os.path.exists(job.output_path):
        os.remove(job.output_path)
    
    # Remove job from memory
    del jobs[job_id]
    
    return {"message": "Files cleaned up successfully"}

@app.get("/status/{job_id}")
async def get_job_status(job_id: str):
    """Get job status"""
    
    if job_id not in jobs:
        raise HTTPException(status_code=404, detail="Job not found")
    
    job = jobs[job_id]
    return {
        "id": job.id,
        "filename": job.filename,
        "status": job.status.value,
        "progress": job.progress,
        "message": job.message,
        "error": job.error,
        "created_at": job.created_at.isoformat()
    }

async def process_video_background(job_id: str):
    """Background task to process video"""
    
    logger.info(f"Starting background processing for job {job_id}")
    job = jobs[job_id]
    
    try:
        job.status = JobStatus.PROCESSING
        job.message = "Starting video processing..."
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
