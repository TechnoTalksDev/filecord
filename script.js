// FileCord Frontend - Connects to FastAPI backend
class FileCordApp {
    constructor() {
        this.apiUrl = 'http://localhost:8000';
        this.currentJobId = null;
        this.eventSource = null;
        
        this.initializeElements();
        this.bindEvents();
    }
    
    initializeElements() {
        // Areas
        this.uploadArea = document.getElementById('upload-area');
        this.loadingArea = document.getElementById('loading-area');
        this.successArea = document.getElementById('success-area');
        this.errorArea = document.getElementById('error-area');
        
        // Upload elements
        this.fileInput = document.getElementById('file-input');
        
        // Loading elements
        this.loadingText = document.getElementById('loading-text');
        this.progressFill = document.getElementById('progress-fill');
        this.progressText = document.getElementById('progress-text');
        
        // Success elements
        this.fileInfo = document.getElementById('file-info');
        this.downloadBtn = document.getElementById('download-btn');
        this.resetBtn = document.getElementById('reset-btn');
        
        // Error elements
        this.errorMessage = document.getElementById('error-message');
        this.retryBtn = document.getElementById('retry-btn');
    }
    
    bindEvents() {
        // Upload area click
        this.uploadArea.addEventListener('click', () => {
            this.fileInput.click();
        });
        
        // File selection
        this.fileInput.addEventListener('change', (e) => {
            if (e.target.files.length > 0) {
                this.handleFileUpload(e.target.files[0]);
            }
        });
        
        // Drag and drop
        this.uploadArea.addEventListener('dragover', (e) => {
            e.preventDefault();
            this.uploadArea.classList.add('drag-over');
        });
        
        this.uploadArea.addEventListener('dragleave', () => {
            this.uploadArea.classList.remove('drag-over');
        });
        
        this.uploadArea.addEventListener('drop', (e) => {
            e.preventDefault();
            this.uploadArea.classList.remove('drag-over');
            
            const files = e.dataTransfer.files;
            if (files.length > 0 && files[0].type.startsWith('video/')) {
                this.handleFileUpload(files[0]);
            } else {
                this.showError('Please upload a valid video file');
            }
        });
        
        // Button events
        this.downloadBtn.addEventListener('click', () => this.downloadVideo());
        this.resetBtn.addEventListener('click', () => this.reset());
        this.retryBtn.addEventListener('click', () => this.reset());
    }
    
    async handleFileUpload(file) {
        console.log('Uploading file:', file.name);
        
        // Validate file
        if (!file.type.startsWith('video/')) {
            this.showError('Please select a video file');
            return;
        }
        
        // Check file size (limit to 100MB for upload)
        if (file.size > 10000 * 1024 * 1024) {
            this.showError('File too large. Please select a file smaller than 10000MB');
            return;
        }
        
        this.showLoading('Uploading video...');
        
        try {
            // Upload file to backend
            const formData = new FormData();
            formData.append('file', file);
            
            const response = await fetch(`${this.apiUrl}/upload`, {
                method: 'POST',
                body: formData
            });
            
            if (!response.ok) {
                const error = await response.json();
                throw new Error(error.detail || 'Upload failed');
            }
            
            const result = await response.json();
            this.currentJobId = result.job_id;
            
            console.log('Upload successful, job ID:', this.currentJobId);
            
            // Start listening for progress updates
            this.startProgressUpdates();
            
        } catch (error) {
            console.error('Upload error:', error);
            this.showError(`Upload failed: ${error.message}`);
        }
    }
    
    startProgressUpdates() {
        if (!this.currentJobId) return;
        
        // Close existing EventSource if any
        if (this.eventSource) {
            this.eventSource.close();
        }
        
        // Create new EventSource for progress updates
        this.eventSource = new EventSource(`${this.apiUrl}/progress/${this.currentJobId}`);
        
        this.eventSource.onmessage = (event) => {
            try {
                const data = JSON.parse(event.data);
                this.handleProgressUpdate(data);
            } catch (error) {
                console.error('Failed to parse progress data:', error);
            }
        };
        
        this.eventSource.onerror = (error) => {
            console.error('SSE error:', error);
            this.eventSource.close();
            
            // Try to get final status
            setTimeout(() => this.checkFinalStatus(), 1000);
        };
    }
    
    handleProgressUpdate(data) {
        console.log('Progress update:', data);
        
        switch (data.status) {
            case 'queued':
                let queueMessage = 'Video queued for processing...';
                if (data.queue_position && data.queue_size) {
                    queueMessage = `Position ${data.queue_position} of ${data.queue_size} in queue`;
                }
                this.updateProgress(0, queueMessage);
                break;
                
            case 'processing':
                this.updateProgress(data.progress, data.message);
                break;
                
            case 'completed':
                this.eventSource?.close();
                this.showSuccess(data.message);
                break;
                
            case 'failed':
                this.eventSource?.close();
                this.showError(data.error || 'Conversion failed');
                break;
        }
    }
    
    async checkFinalStatus() {
        if (!this.currentJobId) return;
        
        try {
            const response = await fetch(`${this.apiUrl}/status/${this.currentJobId}`);
            if (response.ok) {
                const status = await response.json();
                this.handleProgressUpdate(status);
            }
        } catch (error) {
            console.error('Failed to check final status:', error);
            this.showError('Connection lost. Please try again.');
        }
    }
    
    updateProgress(progress, message) {
        this.loadingText.textContent = message;
        this.progressFill.style.width = `${progress}%`;
        this.progressText.textContent = `${Math.round(progress)}%`;
    }
    
    showLoading(message) {
        this.hideAllAreas();
        this.loadingArea.classList.remove('hidden');
        this.updateProgress(0, message);
    }
    
    showSuccess(message) {
        this.hideAllAreas();
        this.successArea.classList.remove('hidden');
        this.fileInfo.textContent = message;
    }
    
    showError(message) {
        this.hideAllAreas();
        this.errorArea.classList.remove('hidden');
        this.errorMessage.textContent = message;
    }
    
    hideAllAreas() {
        this.uploadArea.classList.add('hidden');
        this.loadingArea.classList.add('hidden');
        this.successArea.classList.add('hidden');
        this.errorArea.classList.add('hidden');
    }
    
    async downloadVideo() {
        if (!this.currentJobId) return;
        
        try {
            // Create download link
            const downloadUrl = `${this.apiUrl}/download/${this.currentJobId}`;
            
            // Create temporary link element
            const link = document.createElement('a');
            link.href = downloadUrl;
            link.download = 'converted_video.mp4';
            document.body.appendChild(link);
            link.click();
            document.body.removeChild(link);
            
            console.log('Download started');
            
        } catch (error) {
            console.error('Download error:', error);
            this.showError('Download failed. Please try again.');
        }
    }
    
    async reset() {
        // Clean up EventSource
        if (this.eventSource) {
            this.eventSource.close();
            this.eventSource = null;
        }
        
        // Clean up backend files if we have a job ID
        if (this.currentJobId) {
            try {
                await fetch(`${this.apiUrl}/cleanup/${this.currentJobId}`, {
                    method: 'DELETE'
                });
            } catch (error) {
                console.warn('Cleanup failed:', error);
            }
        }
        
        // Reset state
        this.currentJobId = null;
        this.fileInput.value = '';
        
        // Show upload area
        this.hideAllAreas();
        this.uploadArea.classList.remove('hidden');
    }
}

// Initialize app when DOM is loaded
document.addEventListener('DOMContentLoaded', () => {
    window.fileCordApp = new FileCordApp();
    console.log('FileCord app initialized');
});

// Clean up on page unload
window.addEventListener('beforeunload', () => {
    if (window.fileCordApp) {
        window.fileCordApp.reset();
    }
});
