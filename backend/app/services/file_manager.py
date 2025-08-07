import os
import shutil
from pathlib import Path
from typing import List


class FileManager:
    """Handles file operations and cleanup"""
    
    def __init__(self):
        self.upload_dir = Path("uploads")
        self.output_dir = Path("outputs")
        
        # Ensure directories exist
        self.upload_dir.mkdir(exist_ok=True)
        self.output_dir.mkdir(exist_ok=True)
    
    def save_upload(self, file_content: bytes, filename: str) -> str:
        """Save uploaded file and return path"""
        file_path = self.upload_dir / filename
        
        with open(file_path, "wb") as f:
            f.write(file_content)
        
        return str(file_path)
    
    def cleanup_job_files(self, job_id: str) -> None:
        """Clean up all files associated with a job"""
        # Clean upload files
        for file_path in self.upload_dir.glob(f"{job_id}_*"):
            try:
                file_path.unlink()
            except FileNotFoundError:
                pass
        
        # Clean output files
        for file_path in self.output_dir.glob(f"{job_id}_*"):
            try:
                file_path.unlink()
            except FileNotFoundError:
                pass
    
    def get_file_size_mb(self, file_path: str) -> float:
        """Get file size in megabytes"""
        if not os.path.exists(file_path):
            return 0.0
        return os.path.getsize(file_path) / (1024 * 1024)
    
    def cleanup_old_files(self, max_age_hours: int = 24) -> int:
        """Clean up files older than specified hours, returns count of deleted files"""
        import time
        
        deleted_count = 0
        current_time = time.time()
        
        for directory in [self.upload_dir, self.output_dir]:
            for file_path in directory.iterdir():
                if file_path.is_file():
                    file_age = current_time - file_path.stat().st_mtime
                    if file_age > (max_age_hours * 3600):
                        try:
                            file_path.unlink()
                            deleted_count += 1
                        except FileNotFoundError:
                            pass
        
        return deleted_count
    
    def validate_video_file(self, filename: str, max_size_mb: int = 100) -> bool:
        """Validate uploaded video file"""
        # Check file extension
        valid_extensions = {'.mp4', '.avi', '.mov', '.mkv', '.wmv', '.flv', '.webm', '.m4v'}
        file_ext = Path(filename).suffix.lower()
        
        if file_ext not in valid_extensions:
            return False
        
        return True
