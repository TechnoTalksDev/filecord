## Requirements

- Python 3.8+
- FFmpeg installed on system
- uv package manager

## Installation

1. **Install FFmpeg** (if not already installed):
   - Windows: Download from https://ffmpeg.org/download.html
   - macOS: `brew install ffmpeg`
   - Linux: `sudo apt install ffmpeg`

2. **Install dependencies with uv**:
   ```bash
   cd backend
   uv sync
   ```

## Running the Server

### Development
```bash
cd backend
uv run python run.py
```

The server will start at `http://localhost:8000`

### Production
```bash
cd backend
uv run uvicorn app.main:app --host 0.0.0.0 --port 8000
```

## API Endpoints

### Upload Video
```
POST /upload
Content-Type: multipart/form-data

Body: video file

Response: {"job_id": "uuid", "message": "Upload successful"}
```

### Get Progress (SSE)
```
GET /progress/{job_id}
Accept: text/event-stream

Response: Server-Sent Events with progress updates
```

### Download Converted Video
```
GET /download/{job_id}

Response: Converted video file
```

### Get Job Status
```
GET /status/{job_id}

Response: {"id": "uuid", "status": "completed", "progress": 100, ...}
```

### Cleanup Files
```
DELETE /cleanup/{job_id}

Response: {"message": "Files cleaned up successfully"}
```

## Configuration

Environment variables:
- `HOST`: Server host (default: 0.0.0.0)
- `PORT`: Server port (default: 8000)
- `MAX_FILE_SIZE`: Maximum upload size in MB (default: 100)
- `TARGET_SIZE_MB`: Target output size in MB (default: 10)

## Video Processing

The service uses intelligent compression algorithms:

1. **Analysis**: Probes video metadata (duration, resolution, bitrate)
2. **Calculation**: Determines optimal bitrate for 10MB target
3. **Resolution**: Automatically downscales if needed (1080p → 720p → 480p)
4. **Encoding**: Uses H.264 with optimized settings
5. **Optimization**: Re-encodes if target size not achieved

## Project Structure

```
backend/
├── app/
│   ├── __init__.py
│   ├── main.py              # FastAPI application
│   ├── models.py            # Pydantic models
│   └── services/
│       ├── __init__.py
│       ├── video_processor.py   # Video conversion logic
│       └── file_manager.py      # File operations
├── uploads/                 # Temporary upload storage
├── outputs/                 # Converted video storage
├── pyproject.toml          # uv configuration
├── run.py                  # Development server
└── README.md               # This file
```

## Error Handling

The API handles various error scenarios:
- Invalid file formats
- File size limits
- FFmpeg processing errors
- Network disconnections
- Storage issues



## Development

### Adding New Features

1. **New endpoints**: Add to `app/main.py`
2. **Processing logic**: Extend `services/video_processor.py`
3. **Models**: Define in `app/models.py`

### Testing

```bash
# Run the server
uv run python run.py

# Test upload with curl
curl -X POST "http://localhost:8000/upload" \
     -F "file=@test_video.mp4"
```

## Deployment

For production deployment:

1. Use a process manager (systemd, supervisor)
2. Set up reverse proxy (nginx)
3. Configure environment variables
4. Set up log rotation
5. Monitor disk usage for cleanup

## License

MIT License - see LICENSE file for details
