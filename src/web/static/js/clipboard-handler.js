/**
 * Clipboard Image Handler for Code Horde Chat
 * Handles pasting images from clipboard and uploading them to chat context
 */

class ClipboardImageHandler {
    constructor(chatContainer, apiEndpoint = '/api/upload-image') {
        this.chatContainer = chatContainer;
        this.apiEndpoint = apiEndpoint;
        this.init();
    }

    init() {
        // Listen for paste events on the document
        document.addEventListener('paste', this.handlePaste.bind(this));
        
        // Add visual feedback for drag & drop
        this.setupDragAndDrop();
        
        console.log('Clipboard image handler initialized');
    }

    async handlePaste(event) {
        const items = event.clipboardData?.items;
        if (!items) return;

        // Look for image items in clipboard
        for (const item of items) {
            if (item.type.startsWith('image/')) {
                event.preventDefault();
                
                const file = item.getAsFile();
                if (file) {
                    await this.processImage(file);
                }
                break;
            }
        }
    }

    setupDragAndDrop() {
        const dropZone = this.chatContainer;
        
        dropZone.addEventListener('dragover', (e) => {
            e.preventDefault();
            dropZone.classList.add('drag-over');
        });
        
        dropZone.addEventListener('dragleave', (e) => {
            e.preventDefault();
            dropZone.classList.remove('drag-over');
        });
        
        dropZone.addEventListener('drop', async (e) => {
            e.preventDefault();
            dropZone.classList.remove('drag-over');
            
            const files = Array.from(e.dataTransfer.files);
            for (const file of files) {
                if (file.type.startsWith('image/')) {
                    await this.processImage(file);
                }
            }
        });
    }

    async processImage(file) {
        try {
            // Show loading indicator
            const loadingId = this.showLoadingMessage();
            
            // Create preview
            const preview = await this.createImagePreview(file);
            
            // Upload image
            const uploadResult = await this.uploadImage(file);
            
            if (uploadResult.success) {
                // Add image to chat context
                this.addImageToChat(uploadResult.url, uploadResult.filename, preview);
                
                // Remove loading indicator
                this.removeLoadingMessage(loadingId);
                
                // Show success feedback
                this.showNotification('Image added to chat context', 'success');
            } else {
                throw new Error(uploadResult.error || 'Upload failed');
            }
            
        } catch (error) {
            console.error('Error processing image:', error);
            this.showNotification('Failed to process image: ' + error.message, 'error');
        }
    }

    async createImagePreview(file) {
        return new Promise((resolve) => {
            const reader = new FileReader();
            reader.onload = (e) => resolve(e.target.result);
            reader.readAsDataURL(file);
        });
    }

    async uploadImage(file) {
        const formData = new FormData();
        formData.append('image', file);
        formData.append('context', 'chat');
        formData.append('timestamp', Date.now().toString());

        const response = await fetch(this.apiEndpoint, {
            method: 'POST',
            body: formData,
            headers: {
                'X-Requested-With': 'XMLHttpRequest'
            }
        });

        if (!response.ok) {
            throw new Error(`HTTP ${response.status}: ${response.statusText}`);
        }

        return await response.json();
    }

    addImageToChat(imageUrl, filename, preview) {
        const chatMessages = this.chatContainer.querySelector('.chat-messages');
        if (!chatMessages) return;

        const imageMessage = document.createElement('div');
        imageMessage.className = 'chat-message image-message user-message';
        imageMessage.innerHTML = `
            <div class="message-header">
                <span class="message-type">📷 Image</span>
                <span class="message-time">${new Date().toLocaleTimeString()}</span>
            </div>
            <div class="message-content">
                <div class="image-container">
                    <img src="${imageUrl}" alt="${filename}" class="chat-image" onclick="this.classList.toggle('enlarged')" />
                    <div class="image-info">
                        <span class="filename">${filename}</span>
                        <button class="remove-image" onclick="this.closest('.chat-message').remove()">×</button>
                    </div>
                </div>
                <div class="image-analysis-placeholder">
                    <em>Analyzing image...</em>
                </div>
            </div>
        `;

        chatMessages.appendChild(imageMessage);
        chatMessages.scrollTop = chatMessages.scrollHeight;

        // Trigger image analysis
        this.analyzeImage(imageUrl, imageMessage);
    }

    async analyzeImage(imageUrl, messageElement) {
        try {
            const response = await fetch('/api/analyze-image', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify({
                    image_url: imageUrl,
                    context: 'chat'
                })
            });

            const result = await response.json();
            
            if (result.success) {
                const placeholder = messageElement.querySelector('.image-analysis-placeholder');
                if (placeholder) {
                    placeholder.innerHTML = `
                        <div class="image-analysis">
                            <strong>Analysis:</strong> ${result.description}
                            ${result.objects ? `<br><strong>Objects:</strong> ${result.objects.join(', ')}` : ''}
                            ${result.text ? `<br><strong>Text:</strong> ${result.text}` : ''}
                        </div>
                    `;
                }
            }
        } catch (error) {
            console.error('Image analysis failed:', error);
        }
    }

    showLoadingMessage() {
        const loadingId = 'loading-' + Date.now();
        const chatMessages = this.chatContainer.querySelector('.chat-messages');
        
        const loadingMessage = document.createElement('div');
        loadingMessage.id = loadingId;
        loadingMessage.className = 'chat-message loading-message';
        loadingMessage.innerHTML = `
            <div class="message-content">
                <div class="loading-spinner"></div>
                <span>Processing image...</span>
            </div>
        `;
        
        chatMessages.appendChild(loadingMessage);
        chatMessages.scrollTop = chatMessages.scrollHeight;
        
        return loadingId;
    }

    removeLoadingMessage(loadingId) {
        const loadingMessage = document.getElementById(loadingId);
        if (loadingMessage) {
            loadingMessage.remove();
        }
    }

    showNotification(message, type = 'info') {
        const notification = document.createElement('div');
        notification.className = `notification notification-${type}`;
        notification.textContent = message;
        
        document.body.appendChild(notification);
        
        // Auto-remove after 3 seconds
        setTimeout(() => {
            notification.classList.add('fade-out');
            setTimeout(() => notification.remove(), 300);
        }, 3000);
    }
}

// Auto-initialize when DOM is ready
document.addEventListener('DOMContentLoaded', () => {
    const chatContainer = document.querySelector('.chat-container') || document.body;
    window.clipboardHandler = new ClipboardImageHandler(chatContainer);
});
