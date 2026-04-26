const API_URL = 'https://your-api.onrender.com'; // Change to your API URL
let isProcessing = false;

// DOM elements
const chatMessages = document.getElementById('chat-messages');
const userInput = document.getElementById('user-input');
const sendBtn = document.getElementById('send-btn');
const statusIndicator = document.getElementById('status-indicator');
const statusText = document.getElementById('status-text');

// Event listeners
sendBtn.addEventListener('click', sendMessage);
userInput.addEventListener('keypress', (e) => {
    if (e.key === 'Enter' && !e.shiftKey) {
        e.preventDefault();
        sendMessage();
    }
});

function setStatus(loading, message) {
    if (loading) {
        statusIndicator.classList.add('loading');
        statusText.textContent = message || 'Thinking...';
        sendBtn.disabled = true;
    } else {
        statusIndicator.classList.remove('loading');
        statusText.textContent = message || 'Ready';
        sendBtn.disabled = false;
    }
}

function addMessage(content, isUser) {
    const messageDiv = document.createElement('div');
    messageDiv.className = `message ${isUser ? 'user-message' : 'bot-message'}`;
    
    const contentDiv = document.createElement('div');
    contentDiv.className = 'message-content';
    contentDiv.textContent = content;
    
    messageDiv.appendChild(contentDiv);
    chatMessages.appendChild(messageDiv);
    chatMessages.scrollTop = chatMessages.scrollHeight;
    
    return messageDiv;
}

function addTypingIndicator() {
    const messageDiv = document.createElement('div');
    messageDiv.className = 'message bot-message';
    messageDiv.id = 'typing-indicator';
    
    const typingDiv = document.createElement('div');
    typingDiv.className = 'typing';
    typingDiv.innerHTML = '<span></span><span></span><span></span>';
    
    messageDiv.appendChild(typingDiv);
    chatMessages.appendChild(messageDiv);
    chatMessages.scrollTop = chatMessages.scrollHeight;
    
    return messageDiv;
}

function removeTypingIndicator() {
    const indicator = document.getElementById('typing-indicator');
    if (indicator) indicator.remove();
}

async function sendMessage() {
    if (isProcessing) return;
    
    const message = userInput.value.trim();
    if (!message) return;
    
    isProcessing = true;
    userInput.value = '';
    
    // Add user message to chat
    addMessage(message, true);
    
    // Add typing indicator
    const typingIndicator = addTypingIndicator();
    setStatus(true, 'Generating response...');
    
    try {
        const response = await fetch(`${API_URL}/generate`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                prompt: message,
                max_new_tokens: 512,
                temperature: 0.8,
                top_k: 50,
                do_sample: true
            })
        });
        
        if (!response.ok) throw new Error(`HTTP ${response.status}`);
        
        const data = await response.json();
        removeTypingIndicator();
        addMessage(data.text || 'No response generated', false);
        
    } catch (error) {
        console.error('Error:', error);
        removeTypingIndicator();
        addMessage('Sorry, I encountered an error. Please try again.', false);
        setStatus(false, 'Error');
        setTimeout(() => setStatus(false, 'Ready'), 2000);
    } finally {
        isProcessing = false;
        setStatus(false, 'Ready');
        userInput.focus();
    }
}