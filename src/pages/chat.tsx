import React, { useState, useRef, useEffect } from 'react';
import Layout from '@theme/Layout';
import styles from './chat.module.css';

const ChatPage = () => {
  const [messages, setMessages] = useState([]);
  const [inputValue, setInputValue] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const [selectedText, setSelectedText] = useState(null);
  const messagesEndRef = useRef(null);

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  };

  useEffect(() => {
    scrollToBottom();
  }, [messages]);

  // Function to handle text selection
  useEffect(() => {
    const handleSelection = () => {
      const selectedText = window.getSelection().toString().trim();
      if (selectedText.length > 0) {
        setSelectedText(selectedText);
      }
    };

    document.addEventListener('mouseup', handleSelection);
    return () => {
      document.removeEventListener('mouseup', handleSelection);
    };
  }, []);

  const handleSubmit = async (e) => {
    e.preventDefault();
    if (!inputValue.trim() || isLoading) return;

    // Add user message
    const userMessage = { type: 'user', content: inputValue };
    setMessages(prev => [...prev, userMessage]);
    const currentInput = inputValue;
    const currentSelectedText = selectedText;
    
    setInputValue('');
    setIsLoading(true);

    try {
      // Call backend API
      const response = await fetch('http://localhost:8000/chat', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          question: currentInput,
          selected_text: currentSelectedText || null
        }),
      });

      if (!response.ok) {
        throw new Error('Network response was not ok');
      }

      const data = await response.json();
      
      // Add bot response
      setMessages(prev => [...prev, { type: 'bot', content: data.answer }]);
    } catch (error) {
      console.error('Error:', error);
      setMessages(prev => [...prev, { 
        type: 'bot', 
        content: "Error connecting to server. Please try again." 
      }]);
    } finally {
      setIsLoading(false);
      setSelectedText(null); // Clear selected text after submission
    }
  };

  const clearChat = () => {
    setMessages([]);
  };

  return (
    <Layout title="Physical AI Book Chatbot" description="Ask questions about Physical AI and Humanoid Robotics">
      <div className={styles.chatContainer}>
        <div className={styles.chatHeader}>
          <h1>Physical AI Book Assistant</h1>
          <p>Ask questions about Physical AI and Humanoid Robotics</p>
          {selectedText && (
            <div className={styles.selectedTextNotice}>
              <strong>Selected Text Mode:</strong> {selectedText.substring(0, 100)}...
            </div>
          )}
        </div>

        <div className={styles.chatMessages}>
          {messages.length === 0 ? (
            <div className={styles.welcomeMessage}>
              <h3>Welcome to the Physical AI Book Assistant!</h3>
              <p>Ask me anything about Physical AI and Humanoid Robotics.</p>
              <p>You can also select text on the page to ask questions about specific content.</p>
            </div>
          ) : (
            messages.map((message, index) => (
              <div 
                key={index} 
                className={`${styles.message} ${styles[message.type]}`}
              >
                <div className={styles.messageContent}>
                  {message.content}
                </div>
              </div>
            ))
          )}
          {isLoading && (
            <div className={styles.message + ' ' + styles.bot}>
              <div className={styles.messageContent}>
                <div className={styles.typingIndicator}>
                  <div></div>
                  <div></div>
                  <div></div>
                </div>
              </div>
            </div>
          )}
          <div ref={messagesEndRef} />
        </div>

        <form onSubmit={handleSubmit} className={styles.chatInputForm}>
          <input
            type="text"
            value={inputValue}
            onChange={(e) => setInputValue(e.target.value)}
            placeholder="Ask a question about Physical AI and Humanoid Robotics..."
            className={styles.chatInput}
            disabled={isLoading}
          />
          <button 
            type="submit" 
            className={styles.chatButton}
            disabled={isLoading || !inputValue.trim()}
          >
            {isLoading ? 'Sending...' : 'Send'}
          </button>
        </form>

        <div className={styles.chatActions}>
          <button onClick={clearChat} className={styles.clearButton}>
            Clear Chat
          </button>
        </div>
      </div>
    </Layout>
  );
};

export default ChatPage;