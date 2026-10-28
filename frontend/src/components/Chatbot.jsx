import React, { useState, useRef, useEffect } from 'react';
import './Chatbot.css';

const DEFAULT_SUPPORT_SUGGESTIONS = [
  "What is BrainSightAI?",
  "Supported MRI scan formats",
  "How does neuro analysis pipeline work?",
  "How to contact support?"
];

const DEFAULT_STATUS_SUGGESTIONS = [
  "Show status of my recent jobs",
  "Are there any failed jobs?",
  "Execution summary for today",
  "Why did my last run fail?"
];

// Simple Markdown formatting helper for rich responses
const renderFormattedText = (text) => {
  if (!text) return null;

  // Split text by lines to handle block elements like lists and paragraphs
  const lines = text.split('\n');
  const elements = [];
  let currentList = [];
  let inCodeBlock = false;
  let codeBlockContent = [];

  const flushList = (keyPrefix) => {
    if (currentList.length > 0) {
      elements.push(
        <ul key={`${keyPrefix}-ul`} className="formatted-list">
          {currentList.map((item, idx) => (
            <li key={idx}>{parseInlineFormatting(item)}</li>
          ))}
        </ul>
      );
      currentList = [];
    }
  };

  const parseInlineFormatting = (str) => {
    // Replace **bold**, *italic*, `code`
    const parts = [];
    let keyIdx = 0;

    // Pattern for inline code, bold, links
    const regex = /(\*\*([^*]+)\*\*|\*([^*]+)\*|`([^`]+)`|\[([^\]]+)\]\(([^)]+)\))/g;
    let lastIndex = 0;
    let match;

    while ((match = regex.exec(str)) !== null) {
      // Text before match
      if (match.index > lastIndex) {
        parts.push(str.substring(lastIndex, match.index));
      }

      if (match[2]) {
        // **bold**
        parts.push(<strong key={keyIdx++}>{match[2]}</strong>);
      } else if (match[3]) {
        // *italic*
        parts.push(<em key={keyIdx++}>{match[3]}</em>);
      } else if (match[4]) {
        // `code`
        parts.push(<code key={keyIdx++} className="inline-code">{match[4]}</code>);
      } else if (match[5] && match[6]) {
        // [link](url)
        parts.push(
          <a key={keyIdx++} href={match[6]} target="_blank" rel="noopener noreferrer" className="message-link">
            {match[5]}
          </a>
        );
      }
      lastIndex = regex.lastIndex;
    }

    if (lastIndex < str.length) {
      parts.push(str.substring(lastIndex));
    }

    return parts.length > 0 ? parts : str;
  };

  lines.forEach((line, idx) => {
    const trimmed = line.trim();

    if (trimmed.startsWith('```')) {
      if (inCodeBlock) {
        // End code block
        elements.push(
          <pre key={`code-${idx}`} className="formatted-code-block">
            <code>{codeBlockContent.join('\n')}</code>
          </pre>
        );
        codeBlockContent = [];
        inCodeBlock = false;
      } else {
        flushList(`flush-${idx}`);
        inCodeBlock = true;
      }
      return;
    }

    if (inCodeBlock) {
      codeBlockContent.push(line);
      return;
    }

    if (trimmed.startsWith('- ') || trimmed.startsWith('* ') || trimmed.startsWith('• ')) {
      currentList.push(trimmed.replace(/^[-*•]\s+/, ''));
    } else if (/^\d+\.\s+/.test(trimmed)) {
      currentList.push(trimmed.replace(/^\d+\.\s+/, ''));
    } else {
      flushList(`flush-${idx}`);
      if (trimmed.length > 0) {
        elements.push(
          <p key={`p-${idx}`} className="formatted-paragraph">
            {parseInlineFormatting(line)}
          </p>
        );
      }
    }
  });

  flushList('final-flush');

  if (inCodeBlock && codeBlockContent.length > 0) {
    elements.push(
      <pre key="final-code" className="formatted-code-block">
        <code>{codeBlockContent.join('\n')}</code>
      </pre>
    );
  }

  return elements.length > 0 ? elements : text;
};

const Chatbot = () => {
  const [isOpen, setIsOpen] = useState(true);
  const [chatMode, setChatMode] = useState('unselected'); // 'unselected' | 'patient_status' | 'customer_support'
  const [statusSetupDone, setStatusSetupDone] = useState(false);
  const [hospitalInput, setHospitalInput] = useState('');
  const [patientInput, setPatientInput] = useState('');
  const [hospital, setHospital] = useState('');
  const [patientId, setPatientId] = useState('');
  const [scopeModalOpen, setScopeModalOpen] = useState(false);

  // Claude-style activity state
  const [activityStatus, setActivityStatus] = useState("Thinking...");
  const [elapsedTime, setElapsedTime] = useState(0);
  const [expandedThoughts, setExpandedThoughts] = useState({});

  const [messages, setMessages] = useState([
    {
      id: 1,
      text: "Welcome to BrainSightAI! How may I assist you today?",
      sender: "bot",
      timestamp: new Date(),
      isInitialGreeting: true
    }
  ]);
  const [suggestions, setSuggestions] = useState([]);
  const [input, setInput] = useState('');
  const [isTyping, setIsTyping] = useState(false);
  const [sessionId, setSessionId] = useState(null);
  const [feedbackModal, setFeedbackModal] = useState(null);
  const [feedbackText, setFeedbackText] = useState('');
  const messagesEndRef = useRef(null);

  // API URL
  const API_BASE_URL = process.env.REACT_APP_API_URL || `http://${window.location.hostname}:8000`;

  // Claude-like activity progression timer
  useEffect(() => {
    if (!isTyping) {
      setElapsedTime(0);
      return;
    }

    const startTime = Date.now();
    setActivityStatus("Thinking...");

    const interval = setInterval(() => {
      const elapsedSec = ((Date.now() - startTime) / 1000).toFixed(1);
      setElapsedTime(elapsedSec);

      if (elapsedSec < 1.6) {
        setActivityStatus("Thinking...");
      } else if (elapsedSec < 3.8) {
        setActivityStatus("Searching knowledge base & protocols...");
      } else if (elapsedSec < 6.0) {
        setActivityStatus("Retrieving patient scan records...");
      } else if (elapsedSec < 8.5) {
        setActivityStatus("Diagnosing pipeline execution stages...");
      } else {
        setActivityStatus("Synthesizing final response...");
      }
    }, 100);

    return () => clearInterval(interval);
  }, [isTyping]);

  const toggleThought = (msgId) => {
    setExpandedThoughts(prev => ({
      ...prev,
      [msgId]: !prev[msgId]
    }));
  };

  useEffect(() => {
    if (!sessionId) {
      setSessionId(`session_${Date.now()}`);
    }
  }, [sessionId]);

  useEffect(() => {
    scrollToBottom();
  }, [messages, isTyping, chatMode, statusSetupDone]);

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  };

  // Handle selecting between Patient Status vs Customer Support
  const handleSelectMode = (mode) => {
    setChatMode(mode);

    if (mode === 'customer_support') {
      setStatusSetupDone(true);
      setSuggestions(DEFAULT_SUPPORT_SUGGESTIONS);

      const botMsg = {
        id: Date.now(),
        text: "You're in **Customer Support** mode! Ask me anything about BrainSightAI's neuro-imaging platform, analysis workflows, scan formats, or account details.",
        sender: "bot",
        timestamp: new Date()
      };
      setMessages(prev => [...prev, botMsg]);
    } else if (mode === 'patient_status') {
      setStatusSetupDone(false);
      setSuggestions([]);

      const botMsg = {
        id: Date.now(),
        text: "To check job processing & patient report status, please provide your **Hospital ID** (Required) and **Patient ID** (Optional).",
        sender: "bot",
        timestamp: new Date(),
        isStatusPrompt: true
      };
      setMessages(prev => [...prev, botMsg]);
    }
  };

  // Handle Hospital and Patient ID setup submission
  const handleStatusSubmit = (e) => {
    if (e) e.preventDefault();
    const cleanHospital = hospitalInput.trim();
    const cleanPatient = patientInput.trim();

    if (!cleanHospital) {
      alert("Please enter a valid Hospital ID to continue.");
      return;
    }

    setHospital(cleanHospital);
    setPatientId(cleanPatient);
    setStatusSetupDone(true);
    setScopeModalOpen(false);

    const confirmationMsg = {
      id: Date.now(),
      text: `Active Scope configured:\n- **Hospital ID:** \`${cleanHospital}\`${cleanPatient ? `\n- **Patient ID:** \`${cleanPatient}\`` : '\n- **Patient ID:** *All Patients*'}\n\nWhat would you like to check? You can ask about job statuses, failures, or pipeline execution summaries.`,
      sender: "bot",
      timestamp: new Date()
    };

    setMessages(prev => [...prev, confirmationMsg]);
    setSuggestions(DEFAULT_STATUS_SUGGESTIONS);
  };

  const sendMessage = async (overrideText = null) => {
    const textToSend = overrideText || input;
    if (!textToSend || !textToSend.trim()) return;

    // Check if in status mode but credentials not yet set
    if (chatMode === 'patient_status' && !statusSetupDone && !hospital) {
      alert("Please configure your Hospital ID first before querying patient status.");
      return;
    }

    const userMessage = {
      id: Date.now(),
      text: textToSend,
      sender: "user",
      timestamp: new Date()
    };

    setMessages(prev => [...prev, userMessage]);
    if (!overrideText) setInput('');
    setIsTyping(true);

    try {
      const payload = {
        message: textToSend,
        session_id: sessionId,
        user_id: `user_${Date.now()}`
      };

      // Attach scope if in patient status mode
      if (chatMode === 'patient_status' && hospital) {
        payload.hospital = hospital;
        if (patientId) {
          payload.patient_id = patientId;
        }
      }

      const response = await fetch(`${API_BASE_URL}/api/chat/`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify(payload)
      });

      const data = await response.json();

      if (!response.ok) {
        throw new Error(data.detail || data.error || "Failed to fetch response");
      }

      const botMessage = {
        id: Date.now() + 1,
        text: data.response || "No response received.",
        sender: "bot",
        timestamp: new Date(),
        messageId: data.message_id,
        sources: data.sources || [],
        toolsUsed: data.tools_used || [],
        reactSteps: data.react_steps || [],
        scoped: data.scoped
      };

      setMessages(prev => [...prev, botMessage]);

      // Update suggestions dynamically from backend data
      if (data.suggestions && data.suggestions.length > 0) {
        setSuggestions(data.suggestions);
      } else if (chatMode === 'patient_status') {
        setSuggestions(DEFAULT_STATUS_SUGGESTIONS);
      } else {
        setSuggestions(DEFAULT_SUPPORT_SUGGESTIONS);
      }

    } catch (error) {
      console.error('Error:', error);
      const errorMessage = {
        id: Date.now() + 1,
        text: `Sorry, I encountered an issue: ${error.message || "Trouble connecting to server."}. Please try again.`,
        sender: "bot",
        timestamp: new Date(),
        isError: true
      };
      setMessages(prev => [...prev, errorMessage]);
    } finally {
      setIsTyping(false);
    }
  };

  const handleFeedback = async (messageId, feedbackType) => {
    try {
      await fetch(`${API_BASE_URL}/api/feedback/submit`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          message_id: messageId,
          feedback_type: feedbackType
        })
      });

      setMessages(prev => prev.map(msg =>
        msg.messageId === messageId
          ? { ...msg, feedback: feedbackType }
          : msg
      ));
    } catch (error) {
      console.error('Feedback error:', error);
    }
  };

  const handleDetailedFeedback = async () => {
    if (!feedbackText.trim() || !feedbackModal) return;

    try {
      await fetch(`${API_BASE_URL}/api/feedback/submit`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          message_id: feedbackModal,
          feedback_type: 'detailed_feedback',
          feedback_text: feedbackText
        })
      });

      setMessages(prev => prev.map(msg =>
        msg.messageId === feedbackModal
          ? { ...msg, feedback: 'detailed', feedbackText: feedbackText }
          : msg
      ));

      setFeedbackModal(null);
      setFeedbackText('');
    } catch (error) {
      console.error('Detailed feedback error:', error);
    }
  };

  const handleKeyPress = (e) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      sendMessage();
    }
  };

  const clearChat = () => {
    setChatMode('unselected');
    setStatusSetupDone(false);
    setHospital('');
    setPatientId('');
    setHospitalInput('');
    setPatientInput('');
    setSuggestions([]);
    setMessages([
      {
        id: 1,
        text: "Welcome to BrainSightAI! How may I assist you today?",
        sender: "bot",
        timestamp: new Date(),
        isInitialGreeting: true
      }
    ]);
    setSessionId(`session_${Date.now()}`);
  };

  return (
    <>
      {isOpen && (
        <div className="chatbot-container">
          {/* Header */}
          <div className="chatbot-header">
            <div className="header-content">
              <div className="bot-avatar-small">
                <div className="bot-face">
                  <div className="bot-eyes">
                    <div className="bot-eye"></div>
                    <div className="bot-eye"></div>
                  </div>
                </div>
              </div>
              <div className="header-info">
                <div className="header-title">BrainSightAI Assistant</div>
                <div className="mode-badge-container">
                  {chatMode === 'patient_status' && statusSetupDone && (
                    <span
                      className="mode-badge patient"
                      onClick={() => setScopeModalOpen(true)}
                      title="Click to edit Hospital or Patient ID"
                    >
                      <span className="badge-icon">🩺</span> {hospital} {patientId ? `• ${patientId}` : ''}
                      <span className="badge-edit">✏️</span>
                    </span>
                  )}
                  {chatMode === 'customer_support' && (
                    <span
                      className="mode-badge support"
                      onClick={() => setScopeModalOpen(true)}
                      title="Click to switch mode"
                    >
                      <span className="badge-icon">💬</span> Support Mode
                      <span className="badge-edit">🔄</span>
                    </span>
                  )}
                  {chatMode === 'unselected' && (
                    <span className="status-indicator">
                      <span className="status-dot"></span> Online
                    </span>
                  )}
                </div>
              </div>
            </div>

            <div className="header-actions">
              <button onClick={clearChat} className="clear-btn" title="Reset & Start New Chat">
                <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor">
                  <path d="M3 6h18M19 6v14a2 2 0 01-2 2H7a2 2 0 01-2-2V6m3 0V4a2 2 0 012-2h4a2 2 0 012 2v2" strokeWidth="2" strokeLinecap="round" />
                </svg>
              </button>
              <button onClick={() => setIsOpen(false)} className="close-btn" title="Minimize Chat">
                <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor">
                  <path d="M6 18L18 6M6 6l12 12" strokeWidth="2" strokeLinecap="round" />
                </svg>
              </button>
            </div>
          </div>

          {/* Messages Area */}
          <div className="chatbot-messages">
            {messages.map((message) => (
              <div key={message.id} className={`message-wrapper ${message.sender}`}>
                {message.sender === 'bot' && (
                  <div className="bot-avatar-message">
                    <div className="bot-face-small">
                      <div className="bot-eyes-small">
                        <div className="bot-eye-small"></div>
                        <div className="bot-eye-small"></div>
                      </div>
                    </div>
                  </div>
                )}

                <div className={`message ${message.sender} ${message.isError ? 'error' : ''}`}>
                  <div className="message-text">
                    {renderFormattedText(message.text)}
                  </div>

                  {/* Initial Mode Selection Options */}
                  {message.isInitialGreeting && chatMode === 'unselected' && (
                    <div className="mode-selection-cards">
                      <p className="mode-prompt-label">Please select an option to get started:</p>
                      <button
                        className="mode-card status-card"
                        onClick={() => handleSelectMode('patient_status')}
                      >
                        <div className="card-icon">🩺</div>
                        <div className="card-content">
                          <div className="card-title">Patient & Job Status</div>
                          <div className="card-desc">Check pipeline run states, error logs, and report status.</div>
                        </div>
                      </button>

                      <button
                        className="mode-card support-card"
                        onClick={() => handleSelectMode('customer_support')}
                      >
                        <div className="card-icon">💬</div>
                        <div className="card-content">
                          <div className="card-title">Customer Support & FAQs</div>
                          <div className="card-desc">Ask questions about BrainSightAI platform, docs, and features.</div>
                        </div>
                      </button>
                    </div>
                  )}

                  {/* Inline Status Setup Form */}
                  {message.isStatusPrompt && !statusSetupDone && (
                    <form className="status-setup-form" onSubmit={handleStatusSubmit}>
                      <div className="form-group">
                        <label className="form-label">
                          Hospital ID <span className="required-star">*</span>
                        </label>
                        <input
                          type="text"
                          className="form-input"
                          placeholder="e.g. Apollo-Hospital, 62"
                          value={hospitalInput}
                          onChange={(e) => setHospitalInput(e.target.value)}
                          required
                          autoFocus
                        />
                      </div>

                      <div className="form-group">
                        <label className="form-label">
                          Patient ID <span className="optional-tag">(Optional)</span>
                        </label>
                        <input
                          type="text"
                          className="form-input"
                          placeholder="e.g. BSAI-1024 or leave empty"
                          value={patientInput}
                          onChange={(e) => setPatientInput(e.target.value)}
                        />
                      </div>

                      <div className="form-actions">
                        <button
                          type="submit"
                          className="form-submit-btn"
                          disabled={!hospitalInput.trim()}
                        >
                          Confirm & Start Chat ➔
                        </button>
                        <button
                          type="button"
                          className="form-switch-btn"
                          onClick={() => handleSelectMode('customer_support')}
                        >
                          Switch to Support
                        </button>
                      </div>
                    </form>
                  )}

                  {/* Claude-style Thought Disclosure */}
                  {message.reactSteps && message.reactSteps.length > 0 && (
                    <div className="claude-thought-box">
                      <button
                        type="button"
                        className="claude-thought-toggle"
                        onClick={() => toggleThought(message.id)}
                      >
                        <span className={`claude-chevron ${expandedThoughts[message.id] ? 'open' : ''}`}>›</span>
                        <span className="claude-thought-heading">
                          Thought {(() => {
                            const totalMs = message.reactSteps.reduce((acc, s) => acc + (s.execution_time_ms || 0), 0);
                            return totalMs > 0 ? `for ${(totalMs / 1000).toFixed(1)}s` : '';
                          })()} ({message.reactSteps.length} step{message.reactSteps.length > 1 ? 's' : ''})
                        </span>
                      </button>
                      {expandedThoughts[message.id] && (
                        <div className="claude-thought-drawer">
                          {message.reactSteps.map((step, sIdx) => (
                            <div key={sIdx} className="claude-step-row">
                              <div className="claude-step-header">
                                <span className="claude-step-badge">Step {step.iteration}</span>
                                {step.action && <span className="claude-action-badge">{step.action}</span>}
                                {step.execution_time_ms > 0 && <span className="claude-time-badge">{step.execution_time_ms}ms</span>}
                              </div>
                              {step.thought && <div className="claude-step-body">{step.thought}</div>}
                              {step.observation_summary && (
                                <div className="claude-step-result">
                                  <strong>Observation:</strong> {step.observation_summary}
                                </div>
                              )}
                            </div>
                          ))}
                        </div>
                      )}
                    </div>
                  )}

                  {/* Tool Execution Badges (if any) */}
                  {message.toolsUsed && message.toolsUsed.length > 0 && (
                    <div className="tools-used-container">
                      <div className="tools-header">
                        <span className="tool-gear">⚙️</span> Actions Performed:
                      </div>
                      <div className="tools-list">
                        {message.toolsUsed.map((tool, tIdx) => (
                          <div key={tIdx} className={`tool-pill ${tool.ok ? 'success' : 'failed'}`}>
                            <span className="tool-status-dot"></span>
                            <span className="tool-name">{tool.name}</span>: {tool.summary}
                          </div>
                        ))}
                      </div>
                    </div>
                  )}

                  {/* Knowledge Base Sources */}
                  {message.sources && message.sources.length > 0 && (
                    <div className="sources-container">
                      <div className="sources-title">📚 Verified Sources:</div>
                      <div className="sources-list">
                        {message.sources.map((src, sIdx) => (
                          <div key={sIdx} className="source-item">
                            <span className="source-index">{sIdx + 1}.</span>
                            <span className="source-name">{src.title || "Documentation"}</span>
                          </div>
                        ))}
                      </div>
                    </div>
                  )}

                  {/* Feedback Buttons */}
                  {message.sender === 'bot' && message.messageId && !message.feedback && (
                    <div className="feedback-buttons">
                      <button
                        onClick={() => handleFeedback(message.messageId, 'thumbs_up')}
                        className="feedback-btn"
                        title="Helpful"
                      >
                        <svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor">
                          <path d="M14 9V5a3 3 0 00-3-3l-4 9v11h11.28a2 2 0 002-1.7l1.38-9a2 2 0 00-2-2.3zM7 22H4a2 2 0 01-2-2v-7a2 2 0 012-2h3" strokeWidth="2" strokeLinecap="round" />
                        </svg>
                      </button>
                      <button
                        onClick={() => handleFeedback(message.messageId, 'thumbs_down')}
                        className="feedback-btn"
                        title="Not helpful"
                      >
                        <svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor">
                          <path d="M10 15v4a3 3 0 003 3l4-9V2H5.72a2 2 0 00-2 1.7l-1.38 9a2 2 0 002 2.3zm7-13h2.67A2.31 2.31 0 0122 4v7a2.31 2.31 0 01-2.33 2H17" strokeWidth="2" strokeLinecap="round" />
                        </svg>
                      </button>
                      <button
                        onClick={() => setFeedbackModal(message.messageId)}
                        className="feedback-btn write-feedback"
                        title="Write detailed feedback"
                      >
                        <svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor">
                          <path d="M11 4H4a2 2 0 00-2 2v14a2 2 0 002 2h14a2 2 0 002-2v-7" strokeWidth="2" strokeLinecap="round" />
                          <path d="M18.5 2.5a2.121 2.121 0 013 3L12 15l-4 1 1-4 9.5-9.5z" strokeWidth="2" strokeLinecap="round" />
                        </svg>
                      </button>
                    </div>
                  )}

                  {message.feedback && (
                    <div className="feedback-given">
                      {message.feedback === 'thumbs_up' ? '👍 Helpful' : message.feedback === 'thumbs_down' ? '👎 Not helpful' : '✏️ Detailed feedback'}
                      {message.feedbackText ? ` — "${message.feedbackText}"` : ' submitted'}
                    </div>
                  )}
                </div>
              </div>
            ))}

            {isTyping && (
              <div className="message-wrapper bot">
                <div className="bot-avatar-message">
                  <div className="bot-face-small">
                    <div className="bot-eyes-small">
                      <div className="bot-eye-small"></div>
                      <div className="bot-eye-small"></div>
                    </div>
                  </div>
                </div>
                <div className="message bot typing claude-activity-msg">
                  <div className="claude-activity-row">
                    <span className="claude-activity-spinner"></span>
                    <span className="claude-activity-text">{activityStatus}</span>
                    {elapsedTime > 0 && (
                      <span className="claude-activity-timer">{elapsedTime}s</span>
                    )}
                  </div>
                </div>
              </div>
            )}

            <div ref={messagesEndRef} />
          </div>

          {/* Quick Suggestions Chips (From Past Conversation & DB Data) */}
          {suggestions.length > 0 && !isTyping && (
            <div className="suggestions-container">
              <div className="suggestions-scroll">
                {suggestions.map((sug, sIdx) => (
                  <button
                    key={sIdx}
                    className="suggestion-chip"
                    onClick={() => sendMessage(sug)}
                  >
                    {sug}
                  </button>
                ))}
              </div>
            </div>
          )}

          {/* Input Bar */}
          <div className="chatbot-input">
            <input
              type="text"
              value={input}
              onChange={(e) => setInput(e.target.value)}
              onKeyPress={handleKeyPress}
              placeholder={
                chatMode === 'patient_status' && !statusSetupDone
                  ? "Please fill in the Hospital ID above first..."
                  : chatMode === 'patient_status'
                    ? `Ask about jobs for ${hospital}...`
                    : "Ask a question..."
              }
              disabled={isTyping || (chatMode === 'patient_status' && !statusSetupDone)}
            />
            <button
              onClick={() => sendMessage()}
              disabled={!input.trim() || isTyping || (chatMode === 'patient_status' && !statusSetupDone)}
              className="send-btn"
              title="Send Message"
            >
              <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor">
                <path d="M22 2L11 13M22 2l-7 20-4-9-9-4 20-7z" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" />
              </svg>
            </button>
          </div>
        </div>
      )}

      {/* Scope / Mode Change Modal */}
      {scopeModalOpen && (
        <div className="feedback-modal-overlay">
          <div className="feedback-modal scope-modal">
            <h3>Switch Mode / Update Scope</h3>

            <div className="scope-modal-mode-select">
              <button
                type="button"
                className={`scope-mode-tab ${chatMode === 'patient_status' ? 'active' : ''}`}
                onClick={() => setChatMode('patient_status')}
              >
                🩺 Patient / Job Status
              </button>
              <button
                type="button"
                className={`scope-mode-tab ${chatMode === 'customer_support' ? 'active' : ''}`}
                onClick={() => setChatMode('customer_support')}
              >
                💬 Customer Support
              </button>
            </div>

            {chatMode === 'patient_status' && (
              <div className="scope-modal-form">
                <div className="form-group">
                  <label className="form-label">Hospital ID (Required)</label>
                  <input
                    type="text"
                    className="form-input"
                    value={hospitalInput}
                    onChange={(e) => setHospitalInput(e.target.value)}
                    placeholder="e.g. Apollo-Hospital, 62"
                  />
                </div>
                <div className="form-group">
                  <label className="form-label">Patient ID (Optional)</label>
                  <input
                    type="text"
                    className="form-input"
                    value={patientInput}
                    onChange={(e) => setPatientInput(e.target.value)}
                    placeholder="e.g. BSAI-1024 (Optional)"
                  />
                </div>
              </div>
            )}

            {chatMode === 'customer_support' && (
              <p className="scope-support-desc">
                Customer Support mode lets you ask general questions without tying queries to a specific hospital's database records.
              </p>
            )}

            <div className="feedback-modal-footer">
              <button
                onClick={() => setScopeModalOpen(false)}
                className="modal-btn cancel-btn"
              >
                Cancel
              </button>
              <button
                onClick={() => {
                  if (chatMode === 'patient_status') {
                    handleStatusSubmit();
                  } else {
                    setStatusSetupDone(true);
                    setScopeModalOpen(false);
                    setSuggestions(DEFAULT_SUPPORT_SUGGESTIONS);
                  }
                }}
                className="modal-btn submit-btn"
                disabled={chatMode === 'patient_status' && !hospitalInput.trim()}
              >
                Apply Changes
              </button>
            </div>
          </div>
        </div>
      )}

      {/* Detailed Feedback Modal */}
      {feedbackModal && (
        <div className="feedback-modal-overlay">
          <div className="feedback-modal">
            <h3>Write Your Feedback</h3>
            <textarea
              value={feedbackText}
              onChange={(e) => setFeedbackText(e.target.value)}
              placeholder="Please share your feedback on this response..."
              className="feedback-textarea"
              maxLength={500}
            />
            <div className="feedback-modal-footer">
              <span className="char-count">{feedbackText.length}/500</span>
              <div className="buttons">
                <button
                  onClick={() => {
                    setFeedbackModal(null);
                    setFeedbackText('');
                  }}
                  className="modal-btn cancel-btn"
                >
                  Cancel
                </button>
                <button
                  onClick={handleDetailedFeedback}
                  disabled={!feedbackText.trim()}
                  className="modal-btn submit-btn"
                >
                  Submit Feedback
                </button>
              </div>
            </div>
          </div>
        </div>
      )}

      {/* Floating Toggle Button */}
      <button
        className={`chatbot-toggle ${isOpen ? 'open' : ''}`}
        onClick={() => setIsOpen(!isOpen)}
        title={isOpen ? "Close BrainSightAI Chat" : "Open BrainSightAI Chat"}
      >
        {isOpen ? (
          <svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor">
            <path d="M6 18L18 6M6 6l12 12" strokeWidth="2" strokeLinecap="round" />
          </svg>
        ) : (
          <div className="bot-avatar-toggle">
            <div className="bot-face-toggle">
              <div className="bot-eyes-toggle">
                <div className="bot-eye-toggle"></div>
                <div className="bot-eye-toggle"></div>
              </div>
              <div className="bot-smile"></div>
            </div>
          </div>
        )}
      </button>
    </>
  );
};

export default Chatbot;