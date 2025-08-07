import ffmpeg
import os
import asyncio
import subprocess
import re
import tempfile
import threading
import queue
from typing import AsyncGenerator, Dict, Any, Optional, List, Tuple
from pathlib import Path
import logging
import traceback
from enum import Enum

# Configure logging for video processor
logger = logging.getLogger(__name__)


class HardwareEncoderError(Exception):
    """Exception raised when hardware encoder fails and fallback is needed"""
    pass


class EncoderType(Enum):
    """Available encoder types in priority order (fastest to slowest)"""
    NVENC = "nvenc"      # NVIDIA GPU
    QSV = "qsv"          # Intel Quick Sync Video
    AMF = "amf"          # AMD Advanced Media Framework
    VAAPI = "vaapi"      # Video Acceleration API (Linux)
    SOFTWARE = "software"


class EncoderConfig:
    """Configuration for a specific encoder"""
    def __init__(self, name: str, type: EncoderType, h264_encoder: str, 
                 hevc_encoder: Optional[str] = None, 
                 preset_options: Optional[List[str]] = None,
                 quality_options: Optional[Dict[str, str]] = None,
                 max_bitrate_multiplier: float = 1.0):
        self.name = name
        self.type = type
        self.h264_encoder = h264_encoder
        self.hevc_encoder = hevc_encoder
        self.preset_options = preset_options or []
        self.quality_options = quality_options or {}
        self.max_bitrate_multiplier = max_bitrate_multiplier


class VideoProcessor:
    """Handles video conversion and compression using ffmpeg with GPU acceleration"""

    def __init__(self):
        self.target_size_mb = 10
        self.available_encoders: List[EncoderConfig] = []
        self.preferred_encoder: Optional[EncoderConfig] = None
        self._verify_ffmpeg()
        self._detect_hardware_encoders()
    
    def _verify_ffmpeg(self):
        """Verify that FFmpeg is installed and accessible"""
        try:
            logger.info("Verifying FFmpeg installation...")
            result = subprocess.run(['ffmpeg', '-version'], 
                                  capture_output=True, text=True, timeout=5)
            if result.returncode != 0:
                logger.error("FFmpeg returned non-zero exit code")
                raise Exception("FFmpeg is not properly installed or accessible")
            logger.info("FFmpeg verification successful")
        except (subprocess.TimeoutExpired, subprocess.SubprocessError, FileNotFoundError) as e:
            logger.error(f"FFmpeg verification failed: {str(e)}")
            raise Exception("FFmpeg is not installed or not in PATH. Please install FFmpeg to use video conversion features.")
    
    def _detect_hardware_encoders(self):
        """Detect available hardware encoders and set up encoder configurations"""
        logger.info("Detecting available hardware encoders...")
        
        # Define encoder configurations to test
        encoder_configs = [
            # NVIDIA NVENC (highest priority)
            EncoderConfig(
                name="NVIDIA NVENC",
                type=EncoderType.NVENC,
                h264_encoder="h264_nvenc",
                hevc_encoder="hevc_nvenc",
                preset_options=["fast", "medium", "slow", "hq", "bd", "ll", "llhq", "llhp"],
                quality_options={
                    "preset": "medium",
                    "rc": "vbr_hq",
                    "spatial_aq": "1",
                    "temporal_aq": "1",
                    "rc-lookahead": "20"
                },
                max_bitrate_multiplier=1.2
            ),
            # Intel Quick Sync Video
            EncoderConfig(
                name="Intel Quick Sync Video",
                type=EncoderType.QSV,
                h264_encoder="h264_qsv",
                hevc_encoder="hevc_qsv",
                preset_options=["veryfast", "faster", "fast", "medium", "slow", "slower", "veryslow"],
                quality_options={
                    "preset": "medium",
                    "global_quality": "23",
                    "look_ahead": "1",
                    "look_ahead_depth": "15"
                },
                max_bitrate_multiplier=1.1
            ),
            # AMD AMF
            EncoderConfig(
                name="AMD AMF",
                type=EncoderType.AMF,
                h264_encoder="h264_amf",
                hevc_encoder="hevc_amf",
                preset_options=["speed", "balanced", "quality"],
                quality_options={
                    "quality": "balanced",
                    "rc": "vbr_peak",
                    "qmin": "18",
                    "qmax": "28"
                },
                max_bitrate_multiplier=1.15
            ),
            # VAAPI (Linux mainly)
            EncoderConfig(
                name="VAAPI",
                type=EncoderType.VAAPI,
                h264_encoder="h264_vaapi",
                hevc_encoder="hevc_vaapi",
                preset_options=[],
                quality_options={
                    "rc_mode": "VBR",
                    "global_quality": "25"
                },
                max_bitrate_multiplier=1.1
            )
        ]
        
        # Test each encoder configuration
        working_encoders = []
        for config in encoder_configs:
            logger.info(f"Testing {config.name} ({config.h264_encoder})...")
            if self._test_encoder(config.h264_encoder):
                logger.info(f"✓ Found {config.name} ({config.h264_encoder})")
                working_encoders.append(config)
            else:
                logger.info(f"✗ {config.name} not available ({config.h264_encoder})")  # Upgrade to info for debugging
        
        # Only add working hardware encoders
        self.available_encoders.extend(working_encoders)
        
        # Add software encoder as final fallback (always available)
        software_config = EncoderConfig(
            name="Software (libx264)",
            type=EncoderType.SOFTWARE,
            h264_encoder="libx264",
            hevc_encoder="libx265",
            preset_options=["ultrafast", "superfast", "veryfast", "faster", "fast", "medium", "slow", "slower", "veryslow"],
            quality_options={
                "preset": "medium",
                "crf": "23"
            }
        )
        self.available_encoders.append(software_config)
        
        # Set preferred encoder (first available hardware, fallback to software)
        if self.available_encoders:
            # Prefer hardware encoders over software
            hardware_encoders = [e for e in self.available_encoders if e.type != EncoderType.SOFTWARE]
            if hardware_encoders:
                self.preferred_encoder = hardware_encoders[0]
                logger.info(f"Selected hardware encoder: {self.preferred_encoder.name}")
            else:
                self.preferred_encoder = software_config
                logger.info(f"No hardware encoders available, using: {self.preferred_encoder.name}")
        else:
            logger.warning("No encoders available!")
            
        # Log summary
        hw_count = len([e for e in self.available_encoders if e.type != EncoderType.SOFTWARE])
        logger.info(f"Encoder detection complete: {hw_count} hardware + 1 software encoder available")
    
    def _test_encoder(self, encoder_name: str) -> bool:
        """Test if a specific encoder is available and working with hardware"""
        try:
            # First check if the encoder shows up in the help
            logger.debug(f"Checking if {encoder_name} is available in FFmpeg...")
            result = subprocess.run(
                ['ffmpeg', '-hide_banner', '-h', f'encoder={encoder_name}'],
                capture_output=True, text=True, timeout=5
            )
            if result.returncode != 0 or encoder_name not in result.stdout:
                logger.debug(f"Encoder {encoder_name} not found in FFmpeg help")
                return False
            
            logger.debug(f"Encoder {encoder_name} found in FFmpeg, testing hardware capability...")
            # Now test if the encoder actually works by attempting a real encode
            return self._test_encoder_hardware(encoder_name)
            
        except (subprocess.TimeoutExpired, subprocess.SubprocessError, FileNotFoundError) as e:
            logger.debug(f"Error testing encoder {encoder_name}: {e}")
            return False
    
    def _test_encoder_hardware(self, encoder_name: str) -> bool:
        """Test if encoder actually works with available hardware by doing a real encode test"""
        
        try:
            # Create a minimal test video (1 second, 32x32 pixels)
            with tempfile.NamedTemporaryFile(suffix='.mp4', delete=False) as temp_output:
                temp_output_path = temp_output.name
            
            try:
                # Build test command with minimal settings
                # Use 192x192 resolution to meet NVENC minimum requirements (some GPUs need >=192x192)
                cmd = [
                    'ffmpeg', '-f', 'lavfi', '-i', 'testsrc=duration=0.5:size=192x192:rate=30',
                    '-frames:v', '15', '-c:v', encoder_name
                ]
                
                # Add encoder-specific test parameters
                if 'nvenc' in encoder_name.lower():
                    # NVENC-specific parameters that are more reliable for detection
                    cmd.extend([
                        '-preset', 'fast',           # Fast preset for quick testing
                        '-profile:v', 'main',        # Main profile for compatibility
                        '-b:v', '200k',              # Higher bitrate for stability
                        '-maxrate', '400k',          # Maximum bitrate
                        '-bufsize', '800k',          # Buffer size
                        '-pix_fmt', 'yuv420p'        # Explicit pixel format
                    ])
                elif 'qsv' in encoder_name.lower():
                    cmd.extend(['-preset', 'fast', '-b:v', '200k', '-maxrate', '400k'])
                elif 'amf' in encoder_name.lower():
                    cmd.extend(['-quality', 'speed', '-b:v', '200k'])
                elif 'vaapi' in encoder_name.lower():
                    # VAAPI requires special device setup, skip hardware test for now
                    # We'll fall back to the basic help check for VAAPI
                    logger.debug(f"Skipping hardware test for VAAPI encoder {encoder_name} (requires device setup)")
                    return True
                else:
                    # Software encoder
                    cmd.extend(['-preset', 'ultrafast', '-crf', '30'])
                
                cmd.extend(['-y', '-f', 'null', '-' if os.name != 'nt' else 'NUL'])
                
                # Log the test command for debugging
                logger.debug(f"Testing {encoder_name} with command: {' '.join(cmd)}")
                
                # Run the test with longer timeout for hardware encoders
                timeout_duration = 20 if any(hw in encoder_name.lower() for hw in ['nvenc', 'qsv', 'amf']) else 10
                
                result = subprocess.run(
                    cmd,
                    capture_output=True, text=True, timeout=timeout_duration,
                    # Suppress output on Windows
                    creationflags=subprocess.CREATE_NO_WINDOW if os.name == 'nt' else 0
                )
                
                success = result.returncode == 0
                if not success:
                    # Log the error for debugging but don't spam the logs
                    logger.debug(f"Hardware test failed for {encoder_name}: {result.stderr}")
                    # Check for specific error patterns that indicate hardware unavailability
                    error_output = result.stderr.lower()
                    if any(pattern in error_output for pattern in [
                        'no nvenc capable devices found',
                        'cannot load nvcuda.dll',
                        'cannot load libnvcuda.so',
                        'nvenc initialization failed',
                        'device does not support',
                        'no device available',
                        'hardware acceleration not available',
                        'encoder not found',
                        'unsupported codec',
                        'no supported devices found',
                        'failed to initialize encoder',
                        'error initializing',
                        'not supported',
                        'failed to create',
                        'no compatible device',
                        'codec not supported',
                        'incompatible pixel format',
                        'operation not supported',
                        'driver version is insufficient',
                        'cuda driver version is insufficient',
                        'nvenc encoder not available',
                        'failed to load nvenc',
                        'nvenc dll not found',
                        'nvenc api version',
                        'nvenc device creation failed',
                        'failed to open nvenc codec',
                        'frame dimension less than the minimum supported value',
                        'initializeencoder failed'
                    ]):
                        logger.info(f"Hardware encoder {encoder_name} unavailable: hardware not present or drivers missing")
                    else:
                        logger.debug(f"Hardware encoder {encoder_name} test failed: {result.stderr[:200]}")  # Reduced to debug level
                else:
                    logger.debug(f"Hardware test successful for {encoder_name}")
                
                return success
                
            finally:
                # Clean up temp file
                try:
                    if os.path.exists(temp_output_path):
                        os.unlink(temp_output_path)
                except:
                    pass
                    
        except subprocess.TimeoutExpired:
            logger.debug(f"Hardware test timed out for {encoder_name}")
            return False
        except Exception as e:
            logger.debug(f"Hardware test exception for {encoder_name}: {e}")
            return False
    
    def _is_hardware_encoder_failure(self, stderr_text: str) -> bool:
        """Check if the error indicates a hardware encoder failure"""
        error_patterns = [
            'no nvenc capable devices found',
            'cannot load nvcuda.dll',
            'cannot load libnvcuda.so',
            'nvenc initialization failed',
            'device does not support',
            'no device available',
            'hardware acceleration not available',
            'encoder initialization failed',
            'unsupported codec',
            'no supported devices found',
            'failed to initialize encoder',
            'hardware encoder not available',
            'qsv not available',
            'amf not available',
            'vaapi device creation failed',
            'no vaapi device',
            'failed to initialize hardware encoder'
        ]
        
        stderr_lower = stderr_text.lower()
        return any(pattern in stderr_lower for pattern in error_patterns)
    
    def get_available_encoders(self) -> List[str]:
        """Get list of available encoder names"""
        return [config.name for config in self.available_encoders]
    
    def set_preferred_encoder(self, encoder_name: str) -> bool:
        """Set preferred encoder by name"""
        for config in self.available_encoders:
            if config.name == encoder_name:
                self.preferred_encoder = config
                logger.info(f"Switched to encoder: {encoder_name}")
                return True
        logger.warning(f"Encoder not found: {encoder_name}")
        return False
    
    async def convert_video(
        self, 
        input_path: str, 
        job_id: str, 
        target_size_mb: int = 10
    ) -> AsyncGenerator[Dict[str, Any], None]:
        """
        Convert video to target size with progress updates
        
        Args:
            input_path: Path to input video file
            job_id: Unique job identifier
            target_size_mb: Target file size in MB
            
        Yields:
            Progress updates with status and percentage
        """
        
        logger.info(f"Starting video conversion for job {job_id}")
        logger.info(f"Input: {input_path}, Target size: {target_size_mb}MB")
        
        output_path = f"outputs/{job_id}_converted.mp4"
        
        try:
            # Step 1: Probe video to get metadata
            yield {"progress": 5, "message": "Analyzing video..."}
            logger.info(f"Job {job_id}: Probing video file...")
            
            probe = await self._probe_video(input_path)
            logger.info(f"Job {job_id}: Probe successful, extracting duration...")
            
            duration = self._get_video_duration(probe, input_path)
            logger.info(f"Job {job_id}: Video duration: {duration} seconds")
            
            # Step 2: Calculate optimal encoding settings
            yield {"progress": 10, "message": "Calculating optimal settings..."}
            logger.info(f"Job {job_id}: Calculating encoding settings...")
            
            encoding_settings = self._calculate_encoding_settings(
                probe, target_size_mb, duration
            )
            logger.info(f"Job {job_id}: Encoding settings: {encoding_settings}")
            
            # Step 3: Convert video with progress tracking (with fallback handling)
            yield {"progress": 15, "message": "Starting video conversion..."}
            logger.info(f"Job {job_id}: Starting FFmpeg conversion...")
            
            conversion_successful = False
            fallback_attempted = False
            
            while not conversion_successful:
                try:
                    async for progress in self._convert_with_progress(
                        input_path, output_path, encoding_settings, duration
                    ):
                        # Map conversion progress to 15-95% range
                        mapped_progress = 15 + int(progress * 0.8)
                        yield {
                            "progress": mapped_progress, 
                            "message": f"Converting video... {mapped_progress}%"
                        }
                    
                    conversion_successful = True
                    logger.info(f"Job {job_id}: FFmpeg conversion completed")
                    
                except HardwareEncoderError as e:
                    if not fallback_attempted and self.preferred_encoder.type != EncoderType.SOFTWARE:
                        logger.warning(f"Job {job_id}: Hardware encoder failed, falling back to software encoder")
                        fallback_attempted = True
                        
                        # Switch to software encoder temporarily
                        original_encoder = self.preferred_encoder
                        software_config = next(
                            (config for config in self.available_encoders if config.type == EncoderType.SOFTWARE),
                            None
                        )
                        
                        if software_config:
                            self.preferred_encoder = software_config
                            logger.info(f"Job {job_id}: Switched to {software_config.name} for this conversion")
                            yield {"progress": 15, "message": "Retrying with software encoder..."}
                        else:
                            logger.error(f"Job {job_id}: No software encoder available for fallback")
                            raise Exception("Hardware encoder failed and no software fallback available")
                    else:
                        # Already tried fallback or no fallback available
                        logger.error(f"Job {job_id}: Conversion failed even with fallback encoder")
                        raise Exception(f"Video conversion failed: {str(e)}")
                
                except Exception as e:
                    if "Hardware encoder failed" in str(e) and not fallback_attempted:
                        # Treat as hardware encoder error even if not caught as HardwareEncoderError
                        logger.warning(f"Job {job_id}: Detected hardware encoder failure, attempting fallback")
                        fallback_attempted = True
                        
                        # Switch to software encoder
                        software_config = next(
                            (config for config in self.available_encoders if config.type == EncoderType.SOFTWARE),
                            None
                        )
                        
                        if software_config:
                            self.preferred_encoder = software_config
                            logger.info(f"Job {job_id}: Switched to {software_config.name} for this conversion")
                            yield {"progress": 15, "message": "Retrying with software encoder..."}
                        else:
                            raise Exception("Hardware encoder failed and no software fallback available")
                    else:
                        raise e
            
            # Step 4: Verify output size and optimize if needed
            yield {"progress": 95, "message": "Finalizing..."}
            
            # Restore original encoder if we switched to fallback
            if fallback_attempted and 'original_encoder' in locals():
                logger.info(f"Job {job_id}: Restoring original encoder preference")
                self.preferred_encoder = original_encoder
            
            output_size = os.path.getsize(output_path) / (1024 * 1024)  # MB
            logger.info(f"Job {job_id}: Output file size: {output_size:.2f}MB")
            
            if output_size > target_size_mb * 1.1:  # 10% tolerance
                logger.info(f"Job {job_id}: File too large, optimizing...")
                yield {"progress": 97, "message": "Optimizing file size..."}
                await self._optimize_size(input_path, output_path, target_size_mb, duration)
            
            yield {
                "progress": 100, 
                "message": "Conversion completed!", 
                "output_path": output_path
            }
            logger.info(f"Job {job_id}: Conversion completed successfully")
            
        except Exception as e:
            logger.error(f"Job {job_id}: Video conversion failed with error: {str(e)}")
            logger.error(f"Job {job_id}: Full traceback:")
            logger.error(traceback.format_exc())
            raise Exception(f"Video conversion failed: {str(e)}")
    
    async def _probe_video(self, input_path: str) -> Dict[str, Any]:
        """Probe video file to get metadata"""
        try:
            logger.info(f"Probing video file: {input_path}")
            probe = ffmpeg.probe(input_path)
            logger.info("Video probe successful")
            return probe
        except ffmpeg.Error as e:
            logger.error(f"FFmpeg probe failed: {e.stderr.decode() if e.stderr else str(e)}")
            raise Exception(f"Failed to probe video: {e.stderr.decode() if e.stderr else str(e)}")
        except Exception as e:
            logger.error(f"Unexpected error during probe: {str(e)}")
            raise Exception(f"Failed to probe video: {str(e)}")
    
    def _get_video_duration(self, probe: Dict[str, Any], input_path: str = None) -> float:
        """Extract video duration from probe data with fallback methods"""
        
        logger.info("Extracting video duration from probe data...")
        
        # Method 1: Try to get duration from format
        if 'format' in probe and 'duration' in probe['format']:
            try:
                duration = float(probe['format']['duration'])
                logger.info(f"Duration found in format: {duration} seconds")
                return duration
            except (ValueError, TypeError) as e:
                logger.warning(f"Failed to parse duration from format: {e}")
        
        # Method 2: Try to get duration from video stream
        for i, stream in enumerate(probe.get('streams', [])):
            if stream.get('codec_type') == 'video':
                logger.info(f"Checking video stream {i} for duration...")
                if 'duration' in stream:
                    try:
                        duration = float(stream['duration'])
                        logger.info(f"Duration found in video stream {i}: {duration} seconds")
                        return duration
                    except (ValueError, TypeError) as e:
                        logger.warning(f"Failed to parse duration from video stream {i}: {e}")
                        continue
                        
                # Method 3: Calculate from frame count and frame rate
                if 'nb_frames' in stream and 'r_frame_rate' in stream:
                    try:
                        frame_count = int(stream['nb_frames'])
                        frame_rate_str = stream['r_frame_rate']
                        logger.info(f"Attempting to calculate duration from frames: {frame_count} frames at {frame_rate_str}")
                        
                        # Parse frame rate (e.g., "30/1" or "29.97")
                        if '/' in frame_rate_str:
                            num, den = frame_rate_str.split('/')
                            frame_rate = float(num) / float(den)
                        else:
                            frame_rate = float(frame_rate_str)
                        
                        if frame_rate > 0:
                            duration = frame_count / frame_rate
                            logger.info(f"Calculated duration from frames: {duration} seconds")
                            return duration
                    except (ValueError, TypeError, ZeroDivisionError) as e:
                        logger.warning(f"Failed to calculate duration from frames: {e}")
                        continue
        
        # Method 4: Use ffprobe command directly as last resort
        if input_path:
            logger.info("Trying direct ffprobe command as fallback...")
            try:
                result = subprocess.run([
                    'ffprobe', '-v', 'quiet', '-show_entries', 'format=duration',
                    '-of', 'csv=p=0', input_path
                ], capture_output=True, text=True, timeout=10)
                
                if result.returncode == 0 and result.stdout.strip():
                    duration = float(result.stdout.strip())
                    logger.info(f"Duration from direct ffprobe: {duration} seconds")
                    return duration
                else:
                    logger.warning(f"Direct ffprobe failed: return code {result.returncode}, stderr: {result.stderr}")
            except (subprocess.TimeoutExpired, subprocess.SubprocessError, ValueError) as e:
                logger.warning(f"Direct ffprobe command failed: {e}")
        
        # If all methods fail, log detailed probe data and raise error
        logger.error("All duration extraction methods failed!")
        logger.error(f"Probe data: {probe}")
        raise Exception("Could not determine video duration from any available method")
    
    def _calculate_encoding_settings(
        self, 
        probe: Dict[str, Any], 
        target_size_mb: int, 
        duration: float
    ) -> Dict[str, Any]:
        """Calculate optimal encoding settings based on video metadata"""
        
        video_stream = next(
            (stream for stream in probe['streams'] if stream['codec_type'] == 'video'), 
            None
        )
        
        if not video_stream:
            raise Exception("No video stream found")
        
        # Get original dimensions with fallback
        try:
            width = int(video_stream['width'])
            height = int(video_stream['height'])
        except (KeyError, ValueError, TypeError):
            # Default to common resolution if dimensions not available
            width, height = 1280, 720
            print(f"Warning: Could not determine video dimensions, using default {width}x{height}")
        
        # Calculate target bitrate (reserve 128kbps for audio)
        target_bitrate_kbps = max(
            100,  # Minimum bitrate
            int((target_size_mb * 8192) / duration) - 128  # Target bitrate minus audio
        )
        
        # Determine output resolution
        output_width, output_height = self._calculate_resolution(width, height)
        
        return {
            'video_bitrate': f"{target_bitrate_kbps}k",
            'audio_bitrate': "128k",
            'width': output_width,
            'height': output_height,
            'preset': 'medium',  # Balance between speed and compression
            'crf': None  # We'll use bitrate mode instead
        }
    
    def _calculate_resolution(self, width: int, height: int) -> tuple[int, int]:
        """Calculate appropriate output resolution"""
        
        # Keep aspect ratio
        aspect_ratio = width / height
        
        # Resolution limits based on file size constraint
        if width > 1920 or height > 1080:
            # Downscale to 1080p max
            if aspect_ratio > 16/9:
                return 1920, int(1920 / aspect_ratio)
            else:
                return int(1080 * aspect_ratio), 1080
        elif width > 1280 or height > 720:
            # Consider downscaling to 720p for better compression
            if aspect_ratio > 16/9:
                return 1280, int(1280 / aspect_ratio)
            else:
                return int(720 * aspect_ratio), 720
        else:
            # Keep original resolution if already small
            return width, height
    
    async def _convert_with_progress(
        self, 
        input_path: str, 
        output_path: str, 
        settings: Dict[str, Any], 
        duration: float
    ) -> AsyncGenerator[float, None]:
        """Convert video with progress tracking using hardware acceleration when available"""
        
        # Build ffmpeg command with hardware acceleration
        cmd = self._build_hardware_command(input_path, output_path, settings)
        
        logger.info(f"Starting FFmpeg with {self.preferred_encoder.name}")
        logger.info(f"Command: {' '.join(cmd)}")
        
        # Use threading to run subprocess in a separate thread to avoid Windows asyncio issues
        import threading
        import queue
        
        progress_queue = queue.Queue()
        stderr_lines = []
        
        def run_ffmpeg():
            try:
                process = subprocess.Popen(
                    cmd,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    universal_newlines=True,
                    bufsize=1
                )
                
                logger.info(f"FFmpeg process started with PID: {process.pid}")
                
                # Read stderr line by line for progress
                while True:
                    line = process.stderr.readline()
                    if not line:
                        break
                    
                    stderr_lines.append(line.strip())
                    progress_queue.put(('progress', line.strip()))
                
                process.wait()
                progress_queue.put(('done', process.returncode))
                
            except Exception as e:
                logger.error(f"FFmpeg thread error: {str(e)}")
                progress_queue.put(('error', str(e)))
        
        # Start FFmpeg in a separate thread
        ffmpeg_thread = threading.Thread(target=run_ffmpeg)
        ffmpeg_thread.start()
        
        last_progress = 0.0
        
        try:
            while True:
                # Check for progress updates
                try:
                    msg_type, data = progress_queue.get(timeout=0.1)
                    
                    if msg_type == 'done':
                        returncode = data
                        if returncode != 0:
                            stderr_text = '\n'.join(stderr_lines[-20:])  # Last 20 lines
                            logger.error(f"FFmpeg failed with code {returncode}")
                            logger.error(f"Stderr: {stderr_text}")
                            
                            # Check if this is a hardware encoder failure and try fallback
                            if self._is_hardware_encoder_failure(stderr_text) and self.preferred_encoder.type != EncoderType.SOFTWARE:
                                logger.warning(f"Hardware encoder {self.preferred_encoder.name} failed, attempting fallback to software encoder")
                                raise HardwareEncoderError(f"Hardware encoder failed: {stderr_text}")
                            else:
                                raise Exception(f"FFmpeg conversion failed with code {returncode}. Error: {stderr_text}")
                        
                        logger.info("FFmpeg conversion completed successfully")
                        yield 100.0
                        break
                    
                    elif msg_type == 'error':
                        raise Exception(f"FFmpeg thread error: {data}")
                    
                    elif msg_type == 'progress':
                        line = data
                        # Parse progress from FFmpeg output
                        if 'out_time_ms=' in line:
                            time_ms = line.split('out_time_ms=')[1].split()[0]
                            try:
                                current_time = int(time_ms) / 1000000  # Convert microseconds to seconds
                                progress = min(100.0, (current_time / duration) * 100)
                                if progress > last_progress:
                                    last_progress = progress
                                    yield progress
                            except (ValueError, ZeroDivisionError):
                                pass
                        
                        elif 'progress=end' in line:
                            yield 100.0
                            break
                
                except queue.Empty:
                    await asyncio.sleep(0.1)
                    continue
                
        finally:
            # Ensure thread cleanup
            ffmpeg_thread.join(timeout=5)
            if ffmpeg_thread.is_alive():
                logger.warning("FFmpeg thread did not terminate cleanly")
    
    def _build_hardware_command(self, input_path: str, output_path: str, settings: Dict[str, Any]) -> List[str]:
        """Build FFmpeg command with hardware acceleration parameters"""
        cmd = ['ffmpeg', '-i', input_path]
        
        if not self.preferred_encoder:
            raise Exception("No encoder available")
        
        encoder_config = self.preferred_encoder
        
        # Add encoder-specific parameters
        if encoder_config.type == EncoderType.NVENC:
            cmd.extend([
                '-c:v', encoder_config.h264_encoder,
                '-preset', 'medium',
                '-rc', 'vbr_hq',
                '-b:v', settings['video_bitrate'],
                '-maxrate', f"{int(settings['video_bitrate'].replace('k', '')) * 1.2}k",
                '-bufsize', f"{int(settings['video_bitrate'].replace('k', '')) * 2}k",
                '-spatial_aq', '1',
                '-temporal_aq', '1',
                '-rc-lookahead', '20'
            ])
        elif encoder_config.type == EncoderType.QSV:
            cmd.extend([
                '-c:v', encoder_config.h264_encoder,
                '-preset', 'medium',
                '-global_quality', '23',
                '-b:v', settings['video_bitrate'],
                '-maxrate', f"{int(settings['video_bitrate'].replace('k', '')) * 1.1}k",
                '-bufsize', f"{int(settings['video_bitrate'].replace('k', '')) * 2}k",
                '-look_ahead', '1',
                '-look_ahead_depth', '15'
            ])
        elif encoder_config.type == EncoderType.AMF:
            cmd.extend([
                '-c:v', encoder_config.h264_encoder,
                '-quality', 'balanced',
                '-rc', 'vbr_peak',
                '-b:v', settings['video_bitrate'],
                '-maxrate', f"{int(settings['video_bitrate'].replace('k', '')) * 1.15}k",
                '-bufsize', f"{int(settings['video_bitrate'].replace('k', '')) * 2}k",
                '-qmin', '18',
                '-qmax', '28'
            ])
        elif encoder_config.type == EncoderType.VAAPI:
            cmd.extend([
                '-vaapi_device', '/dev/dri/renderD128',
                '-c:v', encoder_config.h264_encoder,
                '-rc_mode', 'VBR',
                '-b:v', settings['video_bitrate'],
                '-maxrate', f"{int(settings['video_bitrate'].replace('k', '')) * 1.1}k",
                '-global_quality', '25'
            ])
        else:  # SOFTWARE
            cmd.extend([
                '-c:v', encoder_config.h264_encoder,
                '-preset', settings['preset'],
                '-b:v', settings['video_bitrate']
            ])
        
        # Add common parameters
        cmd.extend([
            '-c:a', 'aac',
            '-b:a', settings['audio_bitrate'],
            '-vf', f"scale={settings['width']}:{settings['height']}",
            '-movflags', '+faststart',  # Enable progressive download
            '-y',  # Overwrite output
            '-progress', 'pipe:2',  # Send progress to stderr
            output_path
        ])
        
        return cmd
    
    async def _optimize_size(
        self, 
        input_path: str, 
        output_path: str, 
        target_size_mb: int, 
        duration: float
    ):
        """Optimize file size if initial conversion exceeds target using hardware acceleration"""
        
        logger.info("Starting size optimization with hardware acceleration...")
        
        # Try with lower bitrate (80% of original target)
        lower_bitrate = max(50, int((target_size_mb * 8192 * 0.8) / duration) - 128)
        
        temp_output = f"{output_path}_temp.mp4"
        
        # Use hardware encoder for optimization too
        cmd = ['ffmpeg', '-i', input_path]
        
        if self.preferred_encoder and self.preferred_encoder.type == EncoderType.NVENC:
            cmd.extend([
                '-c:v', self.preferred_encoder.h264_encoder,
                '-preset', 'slow',  # Better compression for NVENC
                '-rc', 'vbr_hq',
                '-b:v', f"{lower_bitrate}k",
                '-maxrate', f"{int(lower_bitrate * 1.2)}k",
                '-bufsize', f"{int(lower_bitrate * 2)}k",
                '-spatial_aq', '1',
                '-temporal_aq', '1'
            ])
        elif self.preferred_encoder and self.preferred_encoder.type == EncoderType.QSV:
            cmd.extend([
                '-c:v', self.preferred_encoder.h264_encoder,
                '-preset', 'slow',
                '-global_quality', '26',  # Higher quality for better compression
                '-b:v', f"{lower_bitrate}k",
                '-maxrate', f"{int(lower_bitrate * 1.1)}k"
            ])
        elif self.preferred_encoder and self.preferred_encoder.type == EncoderType.AMF:
            cmd.extend([
                '-c:v', self.preferred_encoder.h264_encoder,
                '-quality', 'quality',  # Better quality mode
                '-rc', 'vbr_peak',
                '-b:v', f"{lower_bitrate}k",
                '-qmin', '20',
                '-qmax', '30'
            ])
        else:  # SOFTWARE fallback
            cmd.extend([
                '-c:v', 'libx264',
                '-preset', 'slower',  # Better compression
                '-b:v', f"{lower_bitrate}k"
            ])
        
        cmd.extend([
            '-c:a', 'aac',
            '-b:a', '96k',  # Lower audio bitrate
            '-movflags', '+faststart',
            '-y',
            temp_output
        ])
        
        logger.info(f"Optimization command: {' '.join(cmd)}")
        
        # Use threading approach like in _convert_with_progress
        import threading
        import queue
        
        progress_queue = queue.Queue()
        
        def run_optimization():
            try:
                process = subprocess.Popen(
                    cmd,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    universal_newlines=True
                )
                
                stdout, stderr = process.communicate()
                progress_queue.put(('done', process.returncode, stderr))
                
            except Exception as e:
                progress_queue.put(('error', str(e)))
        
        optimization_thread = threading.Thread(target=run_optimization)
        optimization_thread.start()
        
        try:
            while True:
                try:
                    msg_type, *data = progress_queue.get(timeout=0.1)
                    
                    if msg_type == 'done':
                        returncode, stderr = data
                        if returncode != 0:
                            logger.error(f"Optimization failed with code {returncode}")
                            logger.error(f"Stderr: {stderr}")
                            raise Exception(f"Size optimization failed: {stderr}")
                        
                        # Replace original with optimized version
                        os.replace(temp_output, output_path)
                        logger.info("Size optimization completed successfully")
                        break
                    
                    elif msg_type == 'error':
                        raise Exception(f"Optimization error: {data[0]}")
                        
                except queue.Empty:
                    await asyncio.sleep(0.1)
                    continue
                
        finally:
            optimization_thread.join(timeout=30)
            # Clean up temp file if it exists
            if os.path.exists(temp_output):
                try:
                    os.remove(temp_output)
                except:
                    pass
        
        logger.info(f"Final output size after optimization: {os.path.getsize(output_path) / (1024 * 1024):.2f}MB")
    
    def refresh_encoder_detection(self):
        """Re-detect available encoders (useful after driver updates or hardware changes)"""
        logger.info("Refreshing encoder detection...")
        self.available_encoders = []
        self.preferred_encoder = None
        self._detect_hardware_encoders()
        logger.info("Encoder detection refreshed")
    
    def get_encoder_status(self) -> Dict[str, Any]:
        """Get current encoder status and available options"""
        return {
            'preferred_encoder': {
                'name': self.preferred_encoder.name if self.preferred_encoder else 'None',
                'type': self.preferred_encoder.type.value if self.preferred_encoder else 'None',
                'h264_encoder': self.preferred_encoder.h264_encoder if self.preferred_encoder else 'None',
                'hevc_encoder': self.preferred_encoder.hevc_encoder if self.preferred_encoder else 'None'
            },
            'available_encoders': [
                {
                    'name': config.name,
                    'type': config.type.value,
                    'h264_encoder': config.h264_encoder,
                    'hevc_encoder': config.hevc_encoder
                }
                for config in self.available_encoders
            ],
            'total_available': len(self.available_encoders),
            'hardware_available': len([c for c in self.available_encoders if c.type != EncoderType.SOFTWARE])
        }
