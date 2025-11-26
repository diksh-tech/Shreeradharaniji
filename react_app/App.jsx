import React, { useState, useRef, useEffect } from 'react';
import { Send, Plane, Database, Cpu, Zap } from 'lucide-react';
import MessageBubble from './components/MessageBubble';
import Sidebar from './components/Sidebar';
import StatusIndicator from './components/StatusIndicator';
import RouteSelection from './components/RouteSelection';

const API_BASE = import.meta.env.VITE_API_BASE || 'http://localhost:8001';

function App() {
  const [currentSession, setCurrentSession] = useState(null);
  const [sessions, setSessions] = useState([]);
  const [messages, setMessages] = useState([{
    id: 'welcome-' + Date.now(),
    role: 'assistant',
    content: "Hello! 👋 I'm your FlightOps Agent. Ask me anything about flight operations — delays, fuel, passengers, aircraft details, etc.",
    timestamp: new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })
  }]);
  const [input, setInput] = useState('');
  const [loading, setLoading] = useState(false);
  const [agentState, setAgentState] = useState({
    phase: 'idle',
    progress_pct: 0, // ✅ FIXED: Match backend field name
    message: ''
  });
  const [tokenUsage, setTokenUsage] = useState({
    planning: null,
    summarization: null,
    total: null
  });
  const [routeSelection, setRouteSelection] = useState(null);
  const messagesEndRef = useRef(null);

  // Load sessions on app start
  useEffect(() => {
    loadSessions();
  }, []);

  // Load messages when session changes
  useEffect(() => {
    if (currentSession?.id) {
      loadSessionMessages(currentSession.id);
    }
  }, [currentSession]);

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  };

  useEffect(() => {
    scrollToBottom();
  }, [messages]);

  const loadSessions = async () => {
    try {
      const response = await fetch(`${API_BASE}/chat-history`);
      if (response.ok) {
        const data = await response.json();
        setSessions(data.sessions || []);
      }
    } catch (error) {
      console.error('Failed to load sessions:', error);
    }
  };

  const loadSessionMessages = async (sessionId) => {
    try {
      const response = await fetch(`${API_BASE}/sessions/${sessionId}/messages`);
      if (response.ok) {
        const data = await response.json();
        setMessages(data.messages || []);
      }
    } catch (error) {
      console.error('Failed to load session messages:', error);
    }
  };

  const startNewSession = () => {
    const newSessionId = `thread-${Date.now()}`;
    const newSession = {
      id: newSessionId,
      title: 'New Chat',
      created_at: Date.now() / 1000,
      last_activity: Date.now() / 1000,
      message_count: 1,
      last_message: "New conversation started",
      user_query: ""
    };

    setCurrentSession(newSession);
    setMessages([{
      id: 'welcome-' + Date.now(),
      role: 'assistant',
      content: "Hello! 👋 I'm your FlightOps Agent. Ask me anything about flight operations — delays, fuel, passengers, aircraft details, etc.",
      timestamp: new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })
    }]);
    setRouteSelection(null);
    setTokenUsage({ planning: null, summarization: null, total: null });
    
    // Add to sessions list immediately for better UX
    setSessions(prev => [newSession, ...prev]);
  };

  // ✅ NEW: Compute total tokens whenever planning or summarization updates
  useEffect(() => {
    const computeTotalTokens = () => {
      const planning = tokenUsage.planning || {};
      const summarization = tokenUsage.summarization || {};
      
      const total_tokens = (planning.total_tokens || 0) + (summarization.total_tokens || 0);
      const prompt_tokens = (planning.prompt_tokens || 0) + (summarization.prompt_tokens || 0);
      const completion_tokens = (planning.completion_tokens || 0) + (summarization.completion_tokens || 0);

      if (total_tokens > 0) {
        setTokenUsage(prev => ({
          ...prev,
          total: { total_tokens, prompt_tokens, completion_tokens }
        }));
      }
    };

    computeTotalTokens();
  }, [tokenUsage.planning, tokenUsage.summarization]);

  const sendMessage = async () => {
    if (!input.trim() || loading) return;

    // ✅ FIXED: Ensure we have a current session
    const threadId = currentSession?.id || `thread-${Date.now()}`;
    const runId = `run-${Date.now()}`;

    // If no current session, create one
    if (!currentSession) {
      const newSession = {
        id: threadId,
        title: input.substring(0, 50) + (input.length > 50 ? '...' : ''),
        created_at: Date.now() / 1000,
        last_activity: Date.now() / 1000,
        message_count: 1,
        last_message: input,
        user_query: input
      };
      setCurrentSession(newSession);
      setSessions(prev => [newSession, ...prev]);
    }

    const userMessage = {
      id: 'user-' + Date.now(),
      role: 'user',
      content: input,
      timestamp: new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })
    };

    setMessages(prev => [...prev, userMessage]);
    setLoading(true);
    setInput('');
    
    setAgentState({
      phase: 'thinking',
      progress_pct: 5, // ✅ FIXED: Match backend field name
      message: 'Starting analysis...'
    });
    setTokenUsage({ planning: null, summarization: null, total: null });
    setRouteSelection(null);

    const body = {
      thread_id: threadId,
      run_id: runId,
      messages: [{ role: 'user', content: input }],
    };

    try {
      const response = await fetch(`${API_BASE}/agent`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(body),
      });

      if (!response.ok) {
        throw new Error(`HTTP ${response.status}: ${await response.text()}`);
      }

      const reader = response.body.getReader();
      const decoder = new TextDecoder();
      let buffer = '';

      while (true) {
        const { done, value } = await reader.read();
        if (done) break;

        buffer += decoder.decode(value, { stream: true });
        const lines = buffer.split('\n\n');
        buffer = lines.pop() || '';

        for (const line of lines) {
          if (line.startsWith('data: ')) {
            try {
              const event = JSON.parse(line.slice(6));
              handleSSEEvent(event);
            } catch (error) {
              console.warn('Failed to parse SSE event:', error);
            }
          }
        }
      }
    } catch (error) {
      console.error('Error sending message:', error);
      setMessages(prev => [...prev, {
        id: 'error-' + Date.now(),
        role: 'assistant',
        content: `❌ Error: ${error.message}`,
        timestamp: new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })
      }]);
    } finally {
      setLoading(false);
      setAgentState({ phase: 'idle', progress_pct: 0, message: '' });
      loadSessions(); // Refresh sessions list
    }
  };

  const handleSSEEvent = (event) => {
    const timestamp = new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });

    switch (event.type) {
      case 'STATE_UPDATE':
        // ✅ FIXED: Map backend progress_pct to our state
        setAgentState(prev => ({
          ...prev,
          phase: event.state.phase,
          progress_pct: event.state.progress_pct || 0, // ✅ Use progress_pct from backend
          message: event.state.message || ''
        }));
        break;

      case 'TEXT_MESSAGE_CONTENT':
        if (event.message) {
          setMessages(prev => {
            const existingIndex = prev.findIndex(m => m.id === event.message.id);
            if (existingIndex >= 0) {
              const updated = [...prev];
              updated[existingIndex].content += event.message.delta || '';
              updated[existingIndex].timestamp = timestamp;
              return updated;
            } else {
              return [...prev, {
                id: event.message.id,
                role: event.message.role,
                content: event.message.content || event.message.delta || '',
                timestamp
              }];
            }
          });
        }
        break;

      case 'TOOL_CALL_START':
        setMessages(prev => [...prev, {
          id: `tool-${Date.now()}`,
          role: 'system',
          content: `🛠️ Starting ${event.toolCallName}...`,
          timestamp
        }]);
        break;

      case 'TOKEN_USAGE':
        // ✅ FIXED: Store individual usage - total will be computed in useEffect
        setTokenUsage(prev => ({
          ...prev,
          [event.phase]: event.usage
        }));
        break;

      case 'ROUTE_SELECTION_OPTIONS':
        setRouteSelection({
          session_id: event.session_id,
          run_id: event.run_id,
          message: event.message,
          routes: event.routes
        });
        break;

      case 'RUN_FINISHED':
        setAgentState({ phase: 'idle', progress_pct: 0, message: '' });
        break;

      case 'RUN_ERROR':
        setMessages(prev => [...prev, {
          id: `error-${Date.now()}`,
          role: 'assistant',
          content: `❌ ${event.error}`,
          timestamp
        }]);
        setAgentState({ phase: 'idle', progress_pct: 0, message: '' });
        break;

      default:
        console.log('Unhandled event type:', event.type);
    }
  };

  const handleRouteSelection = async (routeIndex) => {
    if (!routeSelection) return;

    try {
      const response = await fetch(`${API_BASE}/sessions/${routeSelection.session_id}/select-route`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ route_index: routeIndex }),
      });

      if (response.ok) {
        const result = await response.json();
        setMessages(prev => [...prev, {
          id: `route-${Date.now()}`,
          role: 'system',
          content: `✅ ${result.message}`,
          timestamp: new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })
        }]);
        setRouteSelection(null);
      } else {
        throw new Error('Route selection failed');
      }
    } catch (error) {
      console.error('Route selection error:', error);
      setMessages(prev => [...prev, {
        id: `error-${Date.now()}`,
        role: 'system',
        content: '❌ Failed to select route',
        timestamp: new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })
      }]);
    }
  };

  const handleKeyPress = (e) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      sendMessage();
    }
  };

  return (
    <div className="flex h-screen bg-gradient-to-br from-blue-50 to-indigo-100">
      {/* Sidebar */}
      <Sidebar
        sessions={sessions}
        currentSession={currentSession}
        onSessionSelect={setCurrentSession}
        onNewSession={startNewSession}
      />

      {/* Main Chat Area */}
      <div className="flex-1 flex flex-col">
        {/* Header */}
        <div className="bg-white/80 backdrop-blur-sm border-b border-gray-200 px-6 py-4">
          <div className="flex items-center justify-between">
            <div className="flex items-center space-x-3">
              <div className="p-2 bg-blue-500 rounded-lg">
                <Plane className="h-6 w-6 text-white" />
              </div>
              <div>
                <h1 className="text-xl font-bold text-gray-800">FlightOps Agent</h1>
                <p className="text-sm text-gray-600">
                  {currentSession ? `Session: ${currentSession.id.slice(-8)}` : 'New Chat'}
                </p>
              </div>
            </div>
            <div className="flex items-center space-x-4 text-sm text-gray-500">
              <div className="flex items-center space-x-1">
                <Database className="h-4 w-4" />
                <span>Redis</span>
              </div>
              <div className="flex items-center space-x-1">
                <Cpu className="h-4 w-4" />
                <span>MCP</span>
              </div>
              <div className="flex items-center space-x-1">
                <Zap className="h-4 w-4" />
                <span>FastAPI</span>
              </div>
            </div>
          </div>
        </div>

        {/* Messages Container */}
        <div className="flex-1 overflow-hidden">
          <div className="h-full flex flex-col">
            {/* Messages */}
            <div className="flex-1 overflow-y-auto px-6 py-4 space-y-4">
              {messages.map((message) => (
                <MessageBubble
                  key={message.id}
                  role={message.role}
                  content={message.content}
                  timestamp={message.timestamp}
                />
              ))}

              {/* Route Selection */}
              {routeSelection && (
                <RouteSelection
                  routes={routeSelection.routes}
                  message={routeSelection.message}
                  onRouteSelect={handleRouteSelection}
                />
              )}

              {/* Status Indicator */}
              {(agentState.phase !== 'idle' || tokenUsage.total) && (
                <StatusIndicator
                  agentState={agentState}
                  tokenUsage={tokenUsage}
                />
              )}

              {/* Loading Indicator */}
              {loading && agentState.phase === 'idle' && (
                <div className="flex justify-start">
                  <div className="assistant-message">
                    <div className="flex space-x-2">
                      <div className="flex space-x-1">
                        <div className="w-2 h-2 bg-gray-400 rounded-full animate-bounce"></div>
                        <div className="w-2 h-2 bg-gray-400 rounded-full animate-bounce" style={{ animationDelay: '0.1s' }}></div>
                        <div className="w-2 h-2 bg-gray-400 rounded-full animate-bounce" style={{ animationDelay: '0.2s' }}></div>
                      </div>
                      <span className="text-gray-500">Thinking...</span>
                    </div>
                  </div>
                </div>
              )}

              <div ref={messagesEndRef} />
            </div>

            {/* Input Area */}
            <div className="border-t border-gray-200 bg-white/50 backdrop-blur-sm p-4">
              <div className="max-w-4xl mx-auto">
                <div className="flex space-x-4">
                  <div className="flex-1">
                    <textarea
                      value={input}
                      onChange={(e) => setInput(e.target.value)}
                      onKeyPress={handleKeyPress}
                      placeholder="Ask about a flight, e.g. 'Why was 6E215 delayed on June 23, 2024?'"
                      className="w-full px-4 py-3 border border-gray-300 rounded-2xl resize-none focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-transparent"
                      rows="2"
                      disabled={loading}
                    />
                  </div>
                  <button
                    onClick={sendMessage}
                    disabled={loading || !input.trim()}
                    className="px-6 py-3 bg-gradient-to-r from-blue-500 to-indigo-600 text-white rounded-2xl font-medium hover:from-blue-600 hover:to-indigo-700 focus:outline-none focus:ring-2 focus:ring-blue-500 focus:ring-offset-2 disabled:opacity-50 disabled:cursor-not-allowed transition-all duration-200 flex items-center space-x-2"
                  >
                    <Send className="h-4 w-4" />
                    <span>Send</span>
                  </button>
                </div>
                <div className="mt-2 text-xs text-gray-500 text-center">
                  Press Enter to send, Shift+Enter for new line
                </div>
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}

export default App;
