# FileCord - Video Converter

A web application that converts any video file to a 10MB MP4 file using a FastAPI backend with FFmpeg processing.

## Features

- **Server-side processing** - Fast, reliable conversion using dedicated FFmpeg backend
- **Any video format** - Supports all formats that FFmpeg can handle
- **10MB target** - Automatically calculates bitrate to achieve approximately 10MB output
- **Real-time progress** - Live streaming progress updates with Server-Sent Events
- **Hardware acceleration** - Automatic detection and use of GPU encoders (NVENC, QSV, AMF)
- **Intelligent fallback** - Automatically falls back to software encoding if hardware fails
- **Minimal UI** - Clean, modern interface with real-time progress indicators
- **Drag & Drop** - Easy file selection with drag and drop support

## How it works

1. Upload any video file by dragging and dropping or clicking to select
2. Backend analyzes your video and calculates optimal settings for 10MB output
3. Converts the video using hardware-accelerated H.264 encoding when available
4. Real-time progress updates stream to your browser via Server-Sent Events
5. Downloads the converted MP4 file when complete

## Technical Details

- **Backend**: FastAPI server with FFmpeg integration
- **Hardware Acceleration**: Supports NVIDIA NVENC, Intel QSV, AMD AMF, and VAAPI
- **Smart Resolution**: Automatically scales down to 720p maximum if needed
- **Bitrate Calculation**: Intelligent target bitrate based on video duration
- **Codecs**: H.264 video codec and AAC audio codec
- **Progress Streaming**: Real-time updates via Server-Sent Events (SSE)
- **Fallback System**: Automatic fallback to software encoding if hardware fails

## Usage

### Frontend
Open `index.html` in a modern web browser to access the upload interface.

### Backend
1. Install uv (if not already installed):
   ```bash
   # On macOS/Linux
   curl -LsSf https://astral.sh/uv/install.sh | sh
   
   # On Windows (PowerShell)
   powershell -c "irm https://astral.sh/uv/install.ps1 | iex"
   
   # Or with pip
   pip install uv
   ```

2. Navigate to the `backend` directory
3. Install dependencies and sync environment:
   ```bash
   uv sync
   ```
4. Start the server:
   ```bash
   uv run python run.py
   # Or directly with uvicorn
   uv run uvicorn app.main:app --reload
   ```
5. The API will be available at `http://localhost:8000`

The frontend automatically connects to the backend for video processing.

## Requirements

### System Requirements
- **FFmpeg**: Must be installed and available in system PATH
- **Python 3.8+**: For the FastAPI backend (uv will manage Python versions if needed)
- **uv**: Fast Python package manager (10-100x faster than pip)
- **GPU (Optional)**: NVIDIA, Intel, or AMD GPU for hardware acceleration

### Hardware Acceleration Support
- **NVIDIA GPUs**: NVENC encoder (requires recent drivers)
- **Intel CPUs**: Quick Sync Video (QSV) on supported processors
- **AMD GPUs**: AMF encoder on supported cards
- **Linux**: VAAPI support on compatible systems

### Browser Requirements
- Modern browser with EventSource (SSE) support
- Chrome, Firefox, Safari, or Edge (recent versions)

## Performance & Features

- **Real-time Progress**: Server-Sent Events provide live updates during conversion
- **Hardware Acceleration**: Up to 10x faster conversion with GPU encoding
- **Intelligent Quality**: Automatic encoder selection for optimal speed/quality balance
- **Robust Processing**: Automatic fallback ensures conversion always completes
- **Optimal File Size**: Target is approximately 10MB, but may vary slightly depending on video content and duration to maintain quality

## Architecture

- **Frontend**: Vanilla JavaScript with Server-Sent Events for real-time updates
- **Backend**: FastAPI with async video processing and progress streaming
- **Processing**: FFmpeg with hardware encoder detection and fallback system
- **Communication**: RESTful API for uploads/downloads, SSE for progress updates
